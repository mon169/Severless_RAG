FROM public.ecr.aws/lambda/python:3.12

# 파이썬 의존성
COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY . ${LAMBDA_TASK_ROOT}

CMD ["handler.main"]
