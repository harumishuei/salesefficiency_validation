import os
import sys
from logging import Logger
from pathlib import Path
import argparse

# This part should be before wdu project import in order to allow to run from everywhere
MY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = str(Path(MY_DIR).parent) + '/wdu'
sys.path.append(os.path.dirname(ROOT_DIR))

from wdu.core.processing_mode import ProcessingMode
from wdu.core.doc_processor import DocProcessor
from wdu.core.output_type import OutputType
from wdu.core.processing_request import ProcessingRequest
from wdu.core.wdu_output import WDUOutput
from wdu.wdu_utils.logger.logger_utils import LoggerUtils
from wdu.wdu_utils.wdu_output_saver import WduOutputSaver


class SingleDocRunner:
    def __init__(self, doc_processor: DocProcessor, logger: Logger):
        self.doc_processor = doc_processor
        self.logger = logger
        self.wdu_output_saver = WduOutputSaver(logger)

    def process_file(self, processing_request : ProcessingRequest, file_path: str, output_folder: str, output_basename_override:str|None = None, separate_folder_4_file = True):
        file_path = os.path.abspath(file_path)
        output_folder = os.path.abspath(output_folder)

        file_name = os.path.basename(file_path)
        _, file_extension = os.path.splitext(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        output_file_base_name = output_basename_override or file_base_name
        if separate_folder_4_file:
            output_folder += f'/{file_base_name}'

        if not Path(output_folder).exists():
            os.makedirs(output_folder)

        wdu_output: WDUOutput = self.doc_processor.process_from_path(file_path, processing_request)
        self.wdu_output_saver.save_to_disk(wdu_output, output_folder, output_file_base_name, processing_request.processing.page_number_based)

        return wdu_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', type=str, required=True,
                        help='path to document to process')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='folder to save the results')
    parser.add_argument('--mode', type=str, default='high_quality',
                        help='type of processing mode express, standard or high_quality')
    parser.add_argument('--output_basename_override', type=str, default=None,
                        help='optional argument to override output base name. Defaults to the stem of the  input file')
    args = parser.parse_args()

    LoggerUtils.clean_default_logger_location()
    logger = LoggerUtils.create_default_logger()

    from wdu.config.wdu_config import WduConfig
    wdu_config = WduConfig()
    wdu_config.models.layout_model.weights_path = "/Users/aa539999/Desktop/wdu2-examples/models/layout_model/artifacts"
    doc_processor = DocProcessor(logger=logger, wdu_config=wdu_config)

    # doc_processor = DocProcessor(logger=logger)
    doc_runner = SingleDocRunner(doc_processor, logger)
    request = ProcessingRequest.default()
    mode = ProcessingMode[args.mode.upper()]
    request.set_processing_mode_params(mode)

    request.requested_outputs = [OutputType.HTML, OutputType.MD, OutputType.WDU_JSON, OutputType.PAGE_IMAGES,
                                 OutputType.TABLES_JSON, OutputType.ASSEMBLY_JSON, OutputType.PLAIN_TEXT]

    doc_runner.process_file(processing_request=request, file_path=args.doc_path, output_folder=args.output_folder, output_basename_override = args.output_basename_override)
    sys.exit(0)