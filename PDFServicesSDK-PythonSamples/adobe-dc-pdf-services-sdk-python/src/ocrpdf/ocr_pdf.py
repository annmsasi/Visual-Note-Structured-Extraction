"""
 Copyright 2024 Adobe
 All Rights Reserved.

 NOTICE: Adobe permits you to use, modify, and distribute this file in
 accordance with the terms of the Adobe license agreement accompanying it.
"""
import json
import fitz  # PyMuPDF
import logging
import os
from datetime import datetime
import re

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.ocr_pdf_job import OCRPDFJob
from adobe.pdfservices.operation.pdfjobs.result.ocr_pdf_result import OCRPDFResult

# Initialize the logger
logging.basicConfig(level=logging.INFO)


#
# This sample illustrates how to perform OCR operation on a PDF file and convert it into a searchable PDF file.
#
# Note that OCR operation on a PDF file results in a PDF file.
#
# Refer to README.md for instructions on how to run the samples.
#
class OcrPDF(object):
    def __init__(self):
        try:
            file = open('src/resources/input.pdf', 'rb')
            input_stream = file.read()
            file.close()
            print("CLIENT ID:", os.getenv("PDF_SERVICES_CLIENT_ID"))
            print("CLIENT SECRET:", os.getenv("PDF_SERVICES_CLIENT_SECRET"))
            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
                client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=input_stream,
                                              mime_type=PDFServicesMediaType.PDF)

            # Creates a new job instance
            ocr_pdf_job = OCRPDFJob(input_asset=input_asset)

            # Submit the job and gets the job result
            location = pdf_services.submit(ocr_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, OCRPDFResult)

            # Get content from the resulting asset(s)
            result_asset: CloudAsset = pdf_services_response.get_result().get_asset()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Creates an output stream and copy stream asset's content to it
            output_file_path = self.create_output_file_path()
            # with open(output_file_path, "wb") as file:
            #     file.write(stream_asset.get_input_stream())
            with open(output_file_path, "wb") as file:
                file.write(stream_asset.get_input_stream())

                # Extract text from the OCR/searchable PDF
                doc = fitz.open(output_file_path)

                pages = []
                full_text = ""

                for i, page in enumerate(doc):
                    page_text = page.get_text()
                    full_text += page_text + "\n"

                    pages.append({
                        "page_number": i + 1,
                        "text": page_text
                    })
                clean_text = re.sub(r'[•■▪▫◆◇◦]+', ' ', full_text)
                clean_text = re.sub(r'[:.]a {4,}', ' ', clean_text)
                clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text)
                clean_text = re.sub(r' +', ' ', clean_text)
                ocr_json = {
                "source_pdf": "src/resources/ocrInput.pdf",
                "ocr_pdf": output_file_path,
                "created_at": datetime.now().isoformat(),
                "full_text": clean_text,
                "pages": pages
            }

                json_output_path = output_file_path.replace(".pdf", ".json")

                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(ocr_json, json_file, indent=2, ensure_ascii=False)

                print(f"Saved OCR PDF to: {output_file_path}")
                print(f"Saved OCR JSON to: {json_output_path}")
        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')


    # Generates a string containing a directory structure and file name for the output file
    @staticmethod
    def create_output_file_path() -> str:
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
        os.makedirs("output/OcrPDF", exist_ok=True)
        return f"output/OcrPDF/ocr{time_stamp}.pdf"


if __name__ == "__main__":
    OcrPDF()
