import os
import subprocess
import io
import unicodedata
import bidi.algorithm  # type: ignore
from PIL import Image
from loguru import logger


class ModelTrainer:
    def __init__(self, model_name: str, base_dir: str, tessdata_dir: str):
        self.model_name = model_name
        self.base_dir = base_dir
        self.tessdata_dir = tessdata_dir
        # self.ground_truth_dir = f"{self.base_dir}/data/{self.model_name}-ground-truth"
        self.ground_truth_dir = f"{self.base_dir}"
        self.finetuned_dir = f"{self.base_dir}/{self.model_name}_finetuned"

    def run_command(self, command):
        log_file_path = os.path.join(
            self.finetuned_dir, f"{self.model_name}_finetuned.log"
        )
        with open(log_file_path, "a") as log_file:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            if process.stdout is None:
                logger.error(f"Failed to execute command: {command}")
                return

            for line in process.stdout:
                print(line, end="")  # Print to console
                log_file.write(line)  # Append to log file
            process.wait()
            if process.returncode != 0:
                logger.error(
                    f"Command failed with return code {process.returncode}: {command}"
                )
            else:
                logger.info(f"Command completed successfully: {command}")

    def prepare_directories(self):
        # os.makedirs(self.ground_truth_dir, exist_ok=True)
        os.makedirs(self.finetuned_dir, exist_ok=True)
        # logger.info(f"Created or verified existence of directory: {self.ground_truth_dir}")
        logger.info(f"Created or verified existence of directory: {self.finetuned_dir}")

    def generate_lstmf_files(self):
        for file in os.listdir(self.ground_truth_dir):
            if file.endswith(".png") or file.endswith(".tif"):
                image_path = os.path.join(self.ground_truth_dir, file)
                lstmf_path = os.path.splitext(image_path)[0]
                logger.info(f"Generating LSTMF file for image: {image_path}")
                self.run_command(
                    f"tesseract {image_path} {lstmf_path} --psm 6 lstm.train"
                )
        lstmf_list_path = f"{self.ground_truth_dir}/all-lstmf-list.txt"
        logger.info(f"Creating LSTMF list file: {lstmf_list_path}")
        self.run_command(
            f"find {self.ground_truth_dir} -name '*.lstmf' > {lstmf_list_path}"
        )
        logger.info(
            f"LSTMF files and list generation completed for model: {self.model_name}"
        )

    def generate_unicharset(self):
        logger.info(
            f"Generating unicharset for ground truth files in: {self.ground_truth_dir}"
        )
        try:
            output_file = os.path.join(self.base_dir, "unicharset")

            self.run_command(
                f"unicharset_extractor --output_unicharset {output_file} {self.ground_truth_dir}/*.gt.txt"
            )
            logger.info(
                f"Unicharset generation completed successfully. Output file: {output_file}"
            )
        except Exception as e:
            logger.error(f"Failed to generate unicharset: {e}")

    def train(
        self, iterations=400, learning_rate=0.0001, error_rate=0.01, remove_old=True
    ):
        gt_path = f"{self.ground_truth_dir}/all-lstmf-list.txt"
        traineddata_path = f"{self.tessdata_dir}/{self.model_name}.traineddata"

        if remove_old:
            logger.info(f"Removing old files in: {self.finetuned_dir}")
            self.run_command(f"rm -rf {self.finetuned_dir}/*")

        logger.info(f"Starting training for model: {self.model_name}")
        try:
            self.run_command(
                f"lstmtraining --model_output {self.finetuned_dir}/{self.model_name} "
                f"--continue_from {self.base_dir}/{self.model_name}.lstm "
                f"--traineddata {traineddata_path} "
                f"--train_listfile {gt_path} "
                f"--max_iterations {iterations} "
                # f"2>&1 | tee -a {self.finetuned_dir}/{self.model_name}_finetuned.log"
            )
            logger.info(f"Training completed successfully for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Training failed for model: {self.model_name}. Error: {e}")

    def convert_checkpoint_to_traineddata(self):
        logger.info(
            f"Converting checkpoint to traineddata for model: {self.model_name}"
        )
        try:
            self.run_command(
                f"lstmtraining --stop_training "
                f"--continue_from {self.finetuned_dir}/{self.model_name}_checkpoint "
                f"--traineddata /usr/share/tessdata/{self.model_name}.traineddata "
                f"--model_output {self.finetuned_dir}/{self.model_name}.traineddata"
            )
            logger.info(
                f"Successfully converted checkpoint to traineddata for model: {self.model_name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to convert checkpoint to traineddata for model: {self.model_name}. Error: {e}"
            )

    def evaluate_model(self):
        gt_path = f"{self.ground_truth_dir}/all-lstmf-list.txt"
        logger.info(f"Evaluating model: {self.model_name}")
        try:
            self.run_command(
                f"lstmeval --model {self.finetuned_dir}/{self.model_name}.traineddata "
                f"--eval_listfile {gt_path} "
                f"--traineddata /usr/share/tessdata/{self.model_name}.traineddata"
            )
            logger.info(
                f"Model evaluation completed successfully for model: {self.model_name}"
            )
        except Exception as e:
            logger.error(
                f"Model evaluation failed for model: {self.model_name}. Error: {e}"
            )

    def generate_box_files(self):
        for file in os.listdir(self.ground_truth_dir):
            if file.endswith(".png") or file.endswith(".tif"):
                image_path = os.path.join(self.ground_truth_dir, file)
                txt_path = os.path.splitext(image_path)[0] + ".gt.txt"
                output_path = os.path.splitext(image_path)[0] + ".box"

                logger.info(f"Processing image: {image_path}")
                try:
                    with open(image_path, "rb") as f:
                        im = Image.open(f)
                        width, height = im.size
                    logger.info(
                        f"Image dimensions (width x height): {width} x {height}"
                    )

                    with io.open(txt_path, "r", encoding="utf-8") as f:
                        lines = f.read().strip().split("\n")
                        if len(lines) != 1:
                            raise ValueError(
                                f"ERROR: {txt_path}: Ground truth text file should contain exactly one line, not {len(lines)}"
                            )
                        line = unicodedata.normalize("NFC", lines[0].strip())

                    if line:
                        line = bidi.algorithm.get_display(line)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write("WordStr 0 0 %d %d 0 #%s\n" % (width, height, line))
                            f.write("\t 0 0 %d %d 0\n" % (width, height))
                        logger.info(f"Generated box file: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")

    def extract_base_lstm(self):
        traineddata_path = f"{self.tessdata_dir}/{self.model_name}.traineddata"
        lstm_output_path = f"{self.base_dir}/{self.model_name}.lstm"

        logger.info(f"Extracting base LSTM for model: {self.model_name}")
        try:
            self.run_command(
                f"combine_tessdata -e {traineddata_path} {lstm_output_path}"
            )
            logger.info(f"Successfully extracted base LSTM to: {lstm_output_path}")
        except Exception as e:
            logger.error(
                f"Failed to extract base LSTM for model: {self.model_name}. Error: {e}"
            )

    def train_model(self):
        self.extract_base_lstm()
        self.prepare_directories()
        self.generate_box_files()
        self.generate_lstmf_files()
        self.generate_unicharset()
        self.train(iterations=1000)
        self.convert_checkpoint_to_traineddata()
        self.evaluate_model()
