# Uses Red Hat Linux
#FROM public.ecr.aws/lambda/python:3.7.2021.05.27.08
FROM public.ecr.aws/lambda/python:3.8


RUN bash -c 'echo -e ${LAMBDA_TASK_ROOT}'

# Copy whole context (minus files defined in .dockerignore)
COPY . ${LAMBDA_TASK_ROOT}

#RUN yum update
RUN yum -y install gcc-c++

#RUN /var/lang/bin/python3.7 -m pip install --upgrade pip

COPY requirements.txt .
# TODO: Maybe no '-t .' needed
RUN pip install -r requirements.txt -t .

#RUN pip uninstall dataclasses -y

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.predict" ]