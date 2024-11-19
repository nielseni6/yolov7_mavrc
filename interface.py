import gradio as gr
import sys
from toy_problem_pgt import toy_problem

if __name__ == "__main__":

    with gr.Blocks(title="toy problem demo", theme=gr.themes.Base()) as demo:
        gr.Markdown(
            """
            # Toy Problem Demo
            This is a demo of the toy problem implementation. 
            """
        )
        with gr.Row() as file_settings:
            
            pgt_coeff = gr.Number(label="PGT Coefficent",info="choose a number between 0.1 and 10",
                                minimum=0.1,maximum=10,value=1,interactive=True,step=1,show_label=True)
            focus_coeff = gr.Number(label="Focus Coefficent",info="Choose a number between 0.1 and 1",
                                minimum=0.1,maximum=1,value=0.2,interactive=True,step=1,show_label=True)
            alpha = gr.Number(label="Alpha",info="Choose a number between 1 and 500",
                                minimum=1,maximum=500,value=400,interactive=True,step=1,show_label=True)
            
            #Target info
            #We are just going to give the user access to the number of bounding boxes, and the xy coords
            num_bb = gr.Number(label="Number of Bounding Boxes",info="Choose a number 0",
                                minimum=0,maximum=0,value=0,interactive=True,step=1,show_label=True)
            x_coord = gr.Number(label="PGT Coefficent",info="Choose a number between 0 and 1",
                                minimum=0,maximum=1,value=0.8,interactive=True,step=1,show_label=True)
            y_coord = gr.Number(label="PGT Coefficent",info="Choose a number between 0 and 1",
                                minimum=0,maximum=1,value=0.76,interactive=True,step=1,show_label=True)
        
        with gr.Row() as outputs:
            output_img = gr.Image(type='filepath',label="Output Image",
                                show_download_button=True,show_share_button=True,interactive=False,visible=True)

        # List of components for clearing
        clear_comp_list = [pgt_coeff, focus_coeff, alpha, num_bb, x_coord, y_coord, output_img]

        # Row for start, clear and demo buttons
        with gr.Row() as buttons:
            start = gr.Button(value="Start")
            clear = gr.ClearButton(value='Clear All',components=clear_comp_list,
                    interactive=True,visible=True)

        # List of gradio components that are input into the run_all method (when start button is clicked)
        run_inputs = [pgt_coeff, focus_coeff, alpha, num_bb, x_coord, y_coord]
        
        # List of gradio components that are output from the run_all method (when start button is clicked)
        run_outputs = [output_img]#TODO- Anything else output]

        start.click(toy_problem, inputs=run_inputs, outputs=run_outputs)

    demo.queue().launch()



            


            

