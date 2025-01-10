import gradio as gr
import sys
from toy_problem_pgt import toy_problem

help_guide = """
## Help Guide

This demo allows you to experiment with the toy problem from the Plausibility Guided Training (PGT) paper.

### Input Parameters:

- **PGT Coefficient**: Choose a number between 0.1 and 10. This determines the emphasis given to the PGT loss function in the training process.
- **Focus Coefficient**: Choose a number between 0.01 and 1. This determines the concentration of pixels around the object that will be rewarded. Higher coefficient results in a more focused reward.
- **X Coord**: Choose a number between 0 and 1. This sets the X coordinate of the target.
- **Y Coord**: Choose a number between 0 and 1. This sets the Y coordinate of the target.

### Outputs:

1. **First 2 images**: Displays the distance regulaization map and the first atrribution map step.
2. **Second 8 images**: Displays each other attribution map steps.
3. **PGT Losses**: Visualizes the plausibility losses over each step.
4. **PGT Scores**: Displays the plausibility scores over each step.

"""


if __name__ == "__main__":

    with gr.Blocks(title="toy problem demo", theme=gr.themes.Base()) as demo:
        gr.Markdown(
            """
            # Toy Problem Demo
            This is a demo of the toy problem implementation. 
            """
        )

        with gr.Accordion("Help", open=False):
            gr.Markdown(help_guide)

        with gr.Row() as file_settings:
            
            pgt_coeff = gr.Number(label="PGT Coefficent",info="choose a number between 0.1 and 10",
                                minimum=0.1,maximum=10,value=1,interactive=True,step=1,show_label=True)
            focus_coeff = gr.Number(label="Focus Coefficent",info="Choose a number between 0.1 and 1",
                                minimum=0.01,maximum=1,value=0.2,interactive=True,step=1,show_label=True)
            
            #TODO - Target info (this is where we can adjust)
            #We are just going to give the user access to the number of bounding boxes, and the xy coords
            # num_bb = gr.Number(label="Number of Bounding Boxes",info="Choose a number",
            #                     minimum=0,maximum=0,value=0,interactive=True,step=1,show_label=True)
            x_coord = gr.Number(label="X Coord",info="Choose a number between 0 and 1",
                                minimum=0,maximum=1,value=0.8,interactive=True,step=1,show_label=True)
            y_coord = gr.Number(label="Y Coord",info="Choose a number between 0 and 1",
                                minimum=0,maximum=1,value=0.76,interactive=True,step=1,show_label=True)
        
        with gr.Row() as outputs:
            output_img1 = gr.Image(type='filepath',label="First 2 images",
                                show_download_button=True,show_share_button=True,interactive=False,visible=True)
            output_img2 = gr.Image(type='filepath',label="9 images",
                                show_download_button=True,show_share_button=True,interactive=False,visible=True, scale=4)
            
        with gr.Row() as outputs_2:    
            output_img3 = gr.Image(type='filepath',label="PGT Losses",
                                show_download_button=True,show_share_button=True,interactive=False,visible=True)
            output_img4 = gr.Image(type='filepath',label="PGT Scores",
                                show_download_button=True,show_share_button=True,interactive=False,visible=True)

        # List of components for clearing
        clear_comp_list = [output_img1, output_img2, output_img3, output_img4]

        # Row for start, clear and demo buttons
        with gr.Row() as buttons:
            start = gr.Button(value="Start")
            clear = gr.ClearButton(value='Clear All',components=clear_comp_list,
                    interactive=True,visible=True)

        # List of gradio components that are input into the run_all method (when start button is clicked)
        run_inputs = [pgt_coeff, focus_coeff, x_coord, y_coord]
        
        # List of gradio components that are output from the run_all method (when start button is clicked)
        run_outputs = [output_img1, output_img2, output_img3, output_img4]

        start.click(toy_problem, inputs=run_inputs, outputs=run_outputs)

    demo.queue().launch()



            


            

