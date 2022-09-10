FROM amazon/aws-lambda-python:3.9

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY trained_model .
RUN pip3 install snscrape --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install pandas --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install simpletransformers --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install numpy --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install torch --target "${LAMBDA_TASK_ROOT}"
CMD [ "lambda_function.lambda_handler" ]