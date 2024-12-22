FROM python:3.9-slim
LABEL org.opencontainers.image.authors="Guillermo E."

# LABEL com.example.vendor="Expert.ai"
ENV TZ CET
RUN echo CET > /etc/TZ && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get update && \
    apt-get install -y build-essential lsb-release curl && \
    groupadd app && useradd -ms /bin/bash -u 1000 -g app app

RUN pip install --upgrade pip
RUN mkdir -p /usr/src
ENV PATH="/opt/app/venv/bin/:/home/app/.local/bin/:${PATH}"
WORKDIR /usr/src

COPY . ./
RUN chown -R app ./ && \
    chgrp -R app ./

RUN su -c 'python3 -m pip install -r requirements.txt' app
USER app

# Find and make all .sh files executable
RUN find /usr/src/ -type f -name "*.sh" -exec chmod +x {} \;

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
