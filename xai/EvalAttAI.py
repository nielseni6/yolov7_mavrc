from plaus_functs import get_gradient
import torch
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from pathlib import Path
import numpy as np
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel

class Perturbation:
    def __init__(self, model, opt, nsteps = 10, epsilon = 0.05):#, augment = False, compute_loss = None, loss_metric = "CIoU"):
        self.model = model
        self.nsteps = nsteps
        self.epsilon = epsilon
        self.opt = opt
        # self.augment = opt.augment
        # self.compute_loss = opt.compute_loss
        # self.loss_metric = opt.loss_metric
        self.stats = [[]] * nsteps
        self.loss = [torch.zeros(3, device=opt.device),]*nsteps
        self.ap, self.ap_class, self.wandb_images = [[]] * nsteps, [[]] * nsteps, [[]] * nsteps
        self.p, self.r = [0.,]*nsteps, [0.,]*nsteps
        self.f1, self.mp = [0.,]*nsteps, [0.,]*nsteps
        self.mr, self.map50, self.map = [0.,]*nsteps, [0.,]*nsteps, [0.,]*nsteps
        self.seen = [0,] * nsteps
        self.t0, self.t1 = [0.,]*nsteps, [0.,]*nsteps
   
    def __init_attr__(self, attr_method = get_gradient, out_num_attr = 1, **kwargs):
        self.attr_method = attr_method
        self.attr_kwargs = kwargs
        self.out_num_attr = out_num_attr

        
    def collect_stats(self, img, targets, paths, shapes):
        # for key, value in self.attr_kwargs.items():
        #     self.key = value
        model = self.model
        opt = self.opt
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        im0 = img.clone().detach().requires_grad_(True)
        
        
        out, train_out = model(im0, augment=opt.augment)  # inference and training outputs
        self.attr = self.attr_method(im0,
                                     grad_wrt=train_out[self.out_num_attr], 
                                     **self.attr_kwargs)
        attr = self.attr.clone().detach()
        im0 = img.clone().detach()
        
        for step_i in range(self.nsteps):
            targets_ = targets.clone().detach()
            img_ = im0 + ((self.epsilon * step_i) * attr)
            
            debug = False
            if debug:
                from plot_functs import imshow
                for i_num in range(len(img_)):
                    imshow(img_[i_num].float(), save_path='figs/img')
                    imshow(self.attr[i_num].float(), save_path='figs/attr')
                    
            img_ = img_.half() if opt.half else img_.float()  # uint8 to fp16/32
            
            with torch.no_grad():
                # Run model
                t = time_synchronized()
                out, train_out = model(img_, augment=opt.augment)  # inference and training outputs
                self.t0[step_i] += time_synchronized() - t
                
                # Compute loss
                if opt.compute_loss:
                    self.loss[step_i] += opt.compute_loss([x.float() for x in train_out], targets_, metric=opt.loss_metric)[1][:3]  # box, obj, cls

                # Run NMS
                targets_[:, 2:] *= torch.Tensor([width, height, width, height]).to(opt.device)  # to pixels
                lb = [targets_[targets_[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
                t = time_synchronized()
                out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)
                self.t1[step_i] += time_synchronized() - t
                
            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets_[targets_[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                self.seen[step_i] += 1

                if len(pred) == 0:
                    if nl:
                        self.stats[step_i].append((torch.zeros(0, opt.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img_[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Append to text file
                if opt.save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(opt.save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # # W&B logging - Media Panel Plots
                # if len(self.wandb_images[step_i]) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                #     if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                #         box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                #                     "class_id": int(cls),
                #                     "box_caption": "%s %.3f" % (names[cls], conf),
                #                     "scores": {"class_score": conf},
                #                     "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                #         boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                #         self.wandb_images[step_i].append(wandb_logger.wandb.Image(img_[si], boxes=boxes, caption=path.name))
                # wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], opt.niou, dtype=torch.bool, device=opt.device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img_[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if opt.plots:
                        opt.confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > opt.iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > opt.iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                self.stats[step_i].append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                print(f"Step {step_i} stats: {self.stats[step_i]}")
            self.total_batches = step_i
        return self.stats
    
    def compute_stats(self):
        opt = self.opt
        model = self.model
        stats_all = self.stats
        nc = opt.nc  # number of classes
        loss = self.loss
        ap, ap_class, wandb_images = self.ap, self.ap_class, self.wandb_images
        p, r = self.p, self.r
        f1, mp = self.f1, self.mp
        mr, map50, map = self.mr, self.map50, self.map
        names=opt.names
        save_dir=opt.save_dir
        plots=opt.plots
        training = opt.training
        results = []
        # Compute statistics
        for step_i in range(self.nsteps):
            stats = stats_all[step_i]
            seen = self.seen[step_i]
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():
                p[step_i], r[step_i], ap[step_i], f1[step_i], ap_class[step_i] = ap_per_class(*stats, plot=plots, v5_metric=opt.v5_metric, save_dir=save_dir, names=names)
                ap50, ap[step_i] = ap[step_i][:, 0], ap[step_i].mean(1)  # AP@0.5, AP@0.5:0.95
                mp[step_i], mr[step_i], map50[step_i], map[step_i] = p[step_i].mean(), r[step_i].mean(), ap50.mean(), ap[step_i].mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
            else:
                nt = torch.zeros(1)

            # Print results
            pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
            print(pf % ('all', seen, nt.sum(), mp[step_i], mr[step_i], map50[step_i], map[step_i]))
        
            # Print results per class
            if (opt.verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
                for i, c in enumerate(ap_class[step_i]):
                    print(pf % (names[c], seen, nt[c], p[step_i][i], r[step_i][i], ap50[i], ap[step_i][i]))

            # Print speeds
            t = tuple(x / seen * 1E3 for x in (self.t0[step_i], self.t1[step_i], self.t0[step_i] + self.t1[step_i])) + (opt.imgsz, opt.imgsz, opt.batch_size)  # tuple
            if not training:
                print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

            # # Plots
            # if plots:
            #     opt.confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            #     if wandb_logger and wandb_logger.wandb:
            #         val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            #         wandb_logger.log({"Validation": val_batches})
            # if wandb_images[step_i]:
            #     wandb_logger.log({"Bounding Box Debugger/Images": wandb_images[step_i]})

            # Return results
            model.float()  # for training
            if not training:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
                print(f"Results saved to {save_dir}{s}")
            maps = np.zeros(nc) + map[step_i]
            for i, c in enumerate(ap_class[step_i]):
                maps[c] = ap[step_i][i]
            rtemp = (mp[step_i], mr[step_i], map50[step_i], map[step_i], *(loss[step_i].cpu() / self.total_batches).tolist()), maps, t
            results.append([rtemp])
            print(f"Step {step_i} results: {rtemp}")
            # loss[step_i] = *(loss[step_i].cpu() / self.total_batches).tolist()
        # return (mp, mr, map50, map, loss), maps, t
        return results, stats_all

    def return_faithfulness_scores(self):
        return