from plaus_functs import get_gradient, normalize_batch, get_plaus_score
import torch
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from pathlib import Path
import numpy as np
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel

# from matplotlib.pyplot import colormaps as cm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from plot_functs import imshow, Subplots
# from plaus_functs import normalize_batch
from plot_functs import normalize_tensor
from utils.loss import ComputeLoss, ComputeLossOTA, ComputePGTLossOTA

def nlist(n, inp=None):
    if inp is None:
        return [[] for _ in range(n)]
    else:
        return [inp for _ in range(n)]

def change_noise_batch(signal_tensors, noise_tensors, desired_snr=10):
    
    scaled_noise_tensors = []
    for i in range(signal_tensors.shape[0]):
        signal_tensor = signal_tensors[i]
        noise_tensor = noise_tensors[i]
        # Ensure tensors have the same shape
        assert signal_tensor.shape == noise_tensor.shape

        # Calculate current SNR
        current_snr = 10 * torch.log10(torch.mean(signal_tensor**2) / torch.mean(noise_tensor**2))

        # Calculate noise scaling factor
        scaling_factor = 10**((current_snr - desired_snr) / 20)

        # Scale the noise tensor
        scaled_noise_tensor = noise_tensor * scaling_factor

        # # Add the scaled noise to the signal tensor
        # noisy_signal_tensor = signal_tensor + scaled_noise_tensor
        
        # # Verify SNR
        # estimated_snr = 10 * torch.log10(torch.mean(torch.pow(signal_tensor, 2)) / torch.mean(torch.pow(scaled_noise_tensor, 2)))
        
        scaled_noise_tensors.append(scaled_noise_tensor)
    return torch.stack(scaled_noise_tensors)

def change_noise_snr(signal_tensor, noise_tensor, desired_snr=10):
    """Changes the SNR of noise in a torch tensor image to a desired value.

    Args:
    signal_tensor: A torch tensor representing the signal image.
    noise_tensor: A torch tensor representing the noise image.
    desired_snr: The desired SNR value (in decibels).

    Returns:
    A torch tensor representing the signal image with the adjusted noise.
    """

    # Ensure tensors have the same shape
    assert signal_tensor.shape == noise_tensor.shape

    # Calculate current SNR
    current_snr = 10 * torch.log10(torch.mean(signal_tensor**2) / torch.mean(noise_tensor**2))

    # Calculate noise scaling factor
    scaling_factor = 10**((current_snr - desired_snr) / 20)

    # Scale the noise tensor
    scaled_noise_tensor = noise_tensor * scaling_factor

    # # Add the scaled noise to the signal tensor
    # noisy_signal_tensor = signal_tensor + scaled_noise_tensor
    
    # # Verify SNR
    # estimated_snr = 10 * torch.log10(torch.mean(torch.pow(signal_tensor, 2)) / torch.mean(torch.pow(scaled_noise_tensor, 2)))

    return scaled_noise_tensor

def plot_im_test(img):
    npimg = np.asarray(img)
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
    save_dir = "figs/test"
    plt.savefig(f"{save_dir}.png")

def overlay_attr(img, mask, colormap: str = "jet", alpha: float = 0.7):
    
    cmap = plt.get_cmap(colormap)
    npmask = np.array(mask.clone().detach().cpu().squeeze(0))
    # cmpmask = ((255 * cmap(npmask)[:, :, :3]).astype(np.uint8)).transpose((2, 0, 1))
    cmpmask = (cmap(npmask)[:, :, :3]).transpose((2, 0, 1))
    overlayed_imgnp = ((alpha * (np.asarray(img.clone().detach().cpu())) + (1 - alpha) * cmpmask))
    overlayed_tensor = torch.tensor(overlayed_imgnp, device=img.device)
    
    return overlayed_tensor

class Perturbation:
    def __init__(self, model, opt, nsteps = 10, desired_snr = 5, start=0):#, augment = False, compute_loss = None, loss_metric = "CIoU"):
        self.model = model
        self.nsteps = nsteps
        self.snr = desired_snr # Value of SNR in dB at the end of the perturbation steps (nsteps)
        self.opt = opt
        # self.augment = opt.augment
        # self.compute_loss = opt.compute_loss
        # self.loss_metric = opt.loss_metric
        self.jdict, self.stats = nlist(nsteps), nlist(nsteps)
        self.loss = nlist(nsteps, torch.zeros(3, device=opt.device))#[torch.zeros(3, device=opt.device)  for _ in range(nsteps)]
        self.ap, self.ap_class, self.wandb_images = nlist(nsteps), nlist(nsteps), nlist(nsteps)
        self.wandb_figs = nlist(nsteps)
        self.wandb_attk, self.wandb_attk_overlay = nlist(nsteps), nlist(nsteps)
        self.wandb_attr_overlay = nlist(nsteps)
        self.wandb_attr = nlist(nsteps)
        self.p, self.r = nlist(nsteps, 0.), nlist(nsteps, 0.)
        self.f1, self.mp = nlist(nsteps, 0.), nlist(nsteps, 0.)
        self.mr, self.map50, self.map = nlist(nsteps, 0.), nlist(nsteps, 0.), nlist(nsteps, 0.)
        self.plaus_list, self.plaus_score_total = nlist(nsteps, 0.), 0.0
        self.num_batches = nlist(nsteps, 0)
        self.seen = nlist(nsteps, 0)
        self.t0, self.t1 = nlist(nsteps, 0.), nlist(nsteps, 0.)
        self.debug1, self.debug2 = False, False
        self.debug3 = False
        self.attr_method = None
        self.start = start
   
    def __init_attr__(self, attr_method = get_gradient, out_num_attr = 1, 
                      torchattacks_used=False, **kwargs):
        self.attr_method = attr_method
        self.attr_kwargs = kwargs
        self.out_num_attr = out_num_attr
        self.torchattacks_used = torchattacks_used

        
    def collect_stats(self, img, targets, paths, shapes, batch_i):
        # for key, value in self.attr_kwargs.items():
        #     self.key = value
        model = self.model
        opt = self.opt
        self.snr_list = []
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        im0 = img.clone().detach().requires_grad_(True)
        
        
        out, train_out = model(im0, augment=opt.augment)  # inference and training outputs
        # self.attr = get_gradient(im0, grad_wrt=train_out[self.out_num_attr], norm=True, keepmean=True, absolute=True, grayscale=True)
        if self.attr_method is not None:
            if self.torchattacks_used:
                self.attk = self.attr_method(im0, targets.clone().detach())
            else:
                self.attk = self.attr_method(im0,
                                        grad_wrt=train_out[self.out_num_attr], 
                                        **self.attr_kwargs)
        else:
            self.attk = torch.zeros_like(im0)
        attk = self.attk
        im0 = img.clone().detach()
        
        overlay_list, imgs_shifted = [[] for i in range(self.nsteps)], [[] for i in range(self.nsteps)]
        attr_list = [[] for i in range(self.nsteps)]
        num_imgs = len(img)
        for step_i in range(self.start, self.nsteps):
            targets_ = targets.clone().detach().requires_grad_(True)
            img_ = img.clone().detach().requires_grad_(True)# * 255.0# + ((self.epsilon * step_i) * attk.clone().detach())
            
            attk_ = attk.clone().detach()
            if not self.torchattacks_used:
                if self.attr_method is not None:
                    attk_ = change_noise_batch(img_, attk, desired_snr=self.snr)
                    try:
                        attk_ *= (step_i / (self.nsteps - 1))
                    except:
                        continue
                noise = attk_.clone().detach() - img.clone().detach()
            else:
                noise = attk_
                    
            # Verify SNR
            avg_snr = 0.0
            for ij in range(len(img)):
                estimated_snr = 10 * torch.log10(torch.mean(img[ij] ** 2) / torch.mean(noise[ij] ** 2))
                avg_snr += estimated_snr.item()
            avg_snr /= len(img)
            self.snr_list.append(round(avg_snr, 2))
            
            if self.torchattacks_used:
                img_ = attk.requires_grad_(True)
            else:
                if step_i != 0:
                    img_ = img_ + attk_
                    img_ = torch.clamp(img_, min=0.0, max=1.0)
            
            if self.debug1 and step_i != 0:
                for i_num in range(len(img_)):
                    imshow(img_[i_num].float(), save_path='figs/img')
                    imshow(self.attk[i_num].float(), save_path='figs/attk')
                    
            img_ = img_.half() if opt.half else img_.float()  # uint8 to fp16/32
            
            ########################### Plausibility and Attribution ###########################
            img_ = img_.requires_grad_(True)
            out, train_out = model(img_, augment=opt.augment)
            if opt.loss_attr:
                # batch_loss, bl_components = opt.compute_loss(train_out, targets, img_)
                wrt = opt.compute_loss(train_out, targets, metric=opt.loss_metric)[0]  # box, obj, cls
            else:
                wrt = train_out[self.out_num_attr]
            attr_grad = get_gradient(img_, 
                                     grad_wrt=wrt, 
                                     norm=True, keepmean=True, 
                                     absolute=True, grayscale=True)
            plaus_score = get_plaus_score(img_, 
                                          targets_out = targets.clone().detach(), 
                                          attr = attr_grad)
            self.plaus_score_total += plaus_score
            self.plaus_list[step_i] += plaus_score
            self.num_batches[step_i] += 1
            
            img_ = img_.detach()
            model.zero_grad()
            # self.plaus_list[step_i].append(plaus_score)
            ####################################################################################
            
            with torch.no_grad():
            # if True:
                # Run model
                t = time_synchronized()
                out, train_out = model(img_, augment=opt.augment)  # inference and training outputs
                # out, train_out = out.clone().detach(), [train_out[io].clone().detach() for io in range(len(train_out))]
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
            
            ####################################################################################################

            # debug = False

            
            im = normalize_batch(im0.clone().detach().float())
            for si in range(len(attk)):
            
                # Convert tensors to PIL Images
                atk_map = attk_[si].clone().detach()
                atk_map = torch.abs(atk_map) # Take absolute values of gradients
                atk_map = torch.sum(atk_map, 0, keepdim=True)
                atk_map = normalize_tensor(atk_map) # Normalize attack maps per image in batch
                img_overlay = (overlay_attr(im[si].clone().detach(), atk_map.clone(), alpha = 0.75))
                
                overlay_list[step_i].append(img_overlay.clone().detach().cpu().numpy())
                imgs_shifted[step_i].append((img_[si].clone().detach().float()).cpu().numpy())# / 255.0).cpu().numpy())
                attr_list[step_i].append(attr_grad[si].clone().detach().cpu().numpy())
                
            if step_i != 0:
                if self.debug2:
                    imshow(img_overlay, save_path='figs/img_overlay')
                    imshow(atk_map.float(), save_path='figs/atk_map')
                    imshow(im[si].float(), save_path='figs/img')
                    print("Saved images in ", 'figs/')
            ####################################################################################################
            
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
                

                # W&B logging - Media Panel Plots
                if len(self.wandb_images[step_i]) < opt.log_imgs and opt.wandb_logger.current_epoch >= 0:  # Check for test operation
                    if opt.wandb_logger.current_epoch % opt.wandb_logger.bbox_interval == 0:
                        box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                    "class_id": int(cls),
                                    "box_caption": "%s %.3f" % (opt.names[cls], conf),
                                    "scores": {"class_score": conf},
                                    "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                        boxes = {"predictions": {"box_data": box_data, "class_labels": opt.names}}  # inference-space
                        self.wandb_images[step_i].append(opt.wandb_logger.wandb.Image(img_[si], boxes=boxes, caption=path.name))
                        self.wandb_attk[step_i].append(opt.wandb_logger.wandb.Image(attk_[si], caption=f'{path.name}_atk'))
                        self.wandb_attr[step_i].append(opt.wandb_logger.wandb.Image(attr_grad[si], caption=f'{path.name}_attr'))
                        img_overlay = (overlay_attr(img_[si].clone().detach(), attr_grad[si].clone(), alpha = 0.75))
                        self.wandb_attr_overlay[step_i].append(opt.wandb_logger.wandb.Image(img_overlay, caption=f'{path.name}_attr_overlay'))
                        # self.images[step_i].append(img_[si].clone().detach())
                        # self.attr[step_i].append(attr_grad[si].clone().detach())
                opt.wandb_logger.log_training_progress(predn, path, opt.names) if opt.wandb_logger and opt.wandb_logger.wandb_run else None
            
                # Append to pycocotools JSON dictionary
                if opt.save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        self.jdict.append({'image_id': image_id,
                                    'category_id': opt.coco91class[int(p[5])] if opt.is_coco else int(p[5]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})
                    
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
                # print(f"Step {step_i} stats: {self.stats[step_i]}")
            self.total_batches = step_i
        
        # if self.debug3:
        if len(self.wandb_figs) < int((opt.log_imgs / 4) + 1) and opt.wandb_logger.current_epoch >= 0:  # Check for test operation
            if opt.wandb_logger.current_epoch % opt.wandb_logger.bbox_interval == 0:
                # rowtitles = [f"Step {i}" for i in range(self.nsteps)]
                figimlist = overlay_list[:2].copy()
                rowtitles = ["Original Image", "Overlayed Image"]
                for i in range(1,len(imgs_shifted)):
                    figimlist.append(imgs_shifted[i])
                    rowtitles.append(f"Perturbation Step {i}")
                overlayfig = Subplots(figsize = (40, 5 + (2 * (len(figimlist) - 3))))
                
                sdir = f'figs/overlayed_images_batch_{batch_i}' # should be specific to each run maybe figs/runs/overla...
                for i in range(len(figimlist)):
                    overlayfig.plot_img_list(figimlist[i], nrows = len(figimlist), 
                                rownum=i, rowtitle=rowtitles[i],# coltitles=[f'Image {ip}' for ip in range(len(overlay_list[i]))], 
                                hold=(i<len(figimlist)-1), savedir=sdir)
                
                imwandb = pil_to_tensor(Image.open(f"{sdir}.png")) / 255.0 # this needs to be fixed, does not allow for multiple runs at same time
                self.wandb_figs.append(opt.wandb_logger.wandb.Image(imwandb, caption=sdir))

            # overlayfig.plot_img_list(imgs_shifted[1])


        
        return self.stats
    
    def compute_stats(self):
        opt = self.opt
        model = self.model
        stats_all = self.stats
        nc = opt.nc  # number of classes
        loss = self.loss
        ap, ap_class, wandb_images = self.ap, self.ap_class, self.wandb_images
        wandb_attk, wandb_attr = self.wandb_attk, self.wandb_attr
        p, r = self.p, self.r
        f1, mp = self.f1, self.mp
        mr, map50, map = self.mr, self.map50, self.map
        names=opt.names
        save_dir=opt.save_dir
        plots=opt.plots
        wandb_logger = opt.wandb_logger
        training = opt.training
        results = []
        # Compute statistics
        for step_i in range(self.start, self.nsteps):
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

            # Plots
            if plots:
                opt.confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
                if wandb_logger and wandb_logger.wandb:
                    val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
                    wandb_logger.log({"Validation": val_batches})
            if wandb_images[step_i]:
                wandb_logger.log({f"Bounding Box Debugger/Images for Perturbation Step {step_i} SNR {self.snr_list[step_i-self.start]} {opt.atk}": wandb_images[step_i]})
            # wandb_attk is not displaying in wandb
            if wandb_attk[step_i]:
                wandb_logger.log({f"Adversarial Noise/Step {step_i} SNR {self.snr_list[step_i-self.start]} {opt.atk}": wandb_attk[step_i]})
            # add attribution map to wandb
            if wandb_attr[step_i]:
                wandb_logger.log({f"Attribution/Step {step_i}": wandb_attr[step_i]})
            if self.wandb_attr_overlay[step_i]:
                wandb_logger.log({f"Attribution Overlay/Step {step_i}": self.wandb_attr_overlay[step_i]})
            # Return results
            model.float()  # for training
            if not training:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
                print(f"Results saved to {save_dir}{s}")
            maps = np.zeros(nc) + map[step_i]
            for i, c in enumerate(ap_class[step_i]):
                maps[c] = ap[step_i][i]
            rtemp = (mp[step_i], mr[step_i], map50[step_i], map[step_i], 
                     *(loss[step_i].cpu() / self.total_batches).tolist(), 
                     (self.plaus_list[step_i]/self.num_batches[step_i])), maps, t
            results.append([rtemp])
            print(f"Step {step_i} results: {rtemp}")
            # loss[step_i] = *(loss[step_i].cpu() / self.total_batches).tolist()
        
        # Log wandb image results
        for i_fig, figwb in enumerate(self.wandb_figs):
            wandb_logger.log({f"Figures/Attk Overlay {i_fig} SNR {self.snr_list[-1]} {opt.atk}": figwb})
        # for i_fig, figwb in enumerate(wandb_images):
        #     wandb_logger.log({f"Bounding Box Debugger/Images for Perturbation Step {i_fig} with SNR {self.snr_list[i_fig]} {opt.atk}": figwb})
        # for i_fig, figwb in enumerate(wandb_attk):
        #     wandb_logger.log({f"Adversarial Noise at Step {i_fig} with SNR {self.snr_list[i_fig]} {opt.atk}": figwb})
        # for i_fig, figwb in enumerate(wandb_attr):
        #     wandb_logger.log({f"Attribution at Step {i_fig}": figwb})        

        # if wandb_attk[step_i-self.start]:
        #     wandb_logger.log({f"Attribution at Step {step_i} with SNR {self.snr_list[step_i-self.start]} {opt.atk}": wandb_attk[step_i-self.start]})

        # return (mp, mr, map50, map, loss), maps, t <- results
        return results, stats_all

    def return_faithfulness_scores(self):
        return