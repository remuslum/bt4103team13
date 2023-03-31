FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN python -m spacy download en_core_web_sm

# Copy function code
COPY ./app ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]