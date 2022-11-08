import PIL.Image
from PIL import Image

import modules.scripts as scripts
import gradio as gr
import os

from modules import images, processing
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "batch alpha-mix"

    # Determines when the script should be shown in the dropdown menu via the
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return is_img2img

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0,
                          label="alpha (0 is raw input, 1 feeds exact copy of last output into input)")
        input_dir = gr.Textbox(label="Input file path", lines=1)
        output_dir = gr.Textbox(label="Input file path", lines=1)
        overwrite = gr.Checkbox(False, label="Overwrite existing files")
        return [alpha, input_dir, output_dir, overwrite]

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.

    def run(self, p: Processed, alpha, input_dir, output_dir, overwrite):

        processing.fix_seed(p)

        # global images
        imgs = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if os.path.isfile(file)]

        save_normally = output_dir == ''

        p.do_not_save_grid = True
        p.do_not_save_samples = not save_normally

        # the last output image in the loop to use when interpolating
        last_img: Image = None
        # used to update the progress bar at the top
        state.job_count = len(imgs) * p.n_iter
        for i, image in enumerate(imgs):
            state.job = f"{i + 1} out of {len(imgs)}"
            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            img = Image.open(image)

            # alpha mix code
            if last_img:
                img = img.convert('RGB').resize(last_img.size)
                # blend must be preformed on exactly the same res and format
                img = PIL.Image.blend(img, last_img, alpha)
                # img.show()

            # this is where the input image is set.
            # Sends a list in case multiple output should be produced from each input
            p.init_images = [img] * p.batch_size
            proc: Processed = process_images(p)
            last_img = proc.images[0].convert('RGB')

            for n, processed_image in enumerate(proc.images):
                filename = os.path.basename(image)

                if n > 0:
                    left, right = os.path.splitext(filename)
                    filename = f"{left}-{n}{right}"

                if not save_normally:
                    processed_image.save(os.path.join(output_dir, filename))

        return proc
