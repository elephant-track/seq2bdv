FROM python:3.12.4-slim

RUN apt update && apt install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo "export PATH=/root/.local/bin:$PATH" >> /root/.profile

CMD /bin/bash
