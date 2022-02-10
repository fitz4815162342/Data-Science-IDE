FROM rocker/rstudio:4.1.2 

# system configuration
ENV CRAN_MIRROR https://cloud.r-project.org
RUN mkdir -p "/home/rstudio/persistent" && chmod -R 777 "/home/rstudio/persistent"
VOLUME /home/rstudio/persistent 
WORKDIR /home/rstudio
COPY ./requirements.txt /home/rstudio/requirements.txt
RUN apt-get update --fix-missing \
	&& apt-get install -y \
		ca-certificates \
		build-essential \
		libprotobuf-dev \
		protobuf-compiler \
		cmake \
    	libglib2.0-0 \
	 	libxext6 \
	   	libsm6  \
	   	libxrender1 \
		libxml2-dev \
		libsndfile1 \
		libx11-dev \
		libatlas-base-dev \
		libgtk-3-dev \
		libboost-python-dev \
		ffmpeg \
		python3-audioread \
		libssl-dev \
		openssl \
		python3-openssl \
		pkg-config 		

# install R packages
RUN install2.r --repos ${CRAN_MIRROR}\
		Rcpp \
		devtools \
		roxygen2 \
		knitr \
		rmarkdown \
		yaml \
		reticulate \
		ggplot2 \
		dplyr \
		leaflet \
		shiny \
		shinydashboard \
		openxlsx \
		RColorBrewer \
		rpart \
		gbm \
		plotly \
		readr \
		magrittr \
		DT \
		future \
		stringr \
		docstring \
		imager

# install python and packages
RUN apt-get -y install python3 python3-venv python3-pip
RUN python3 -m venv /home/rstudio/venv
RUN /home/rstudio/venv/bin/pip3 install --upgrade pip setuptools wheel
RUN /home/rstudio/venv/bin/pip3 install -r /home/rstudio/requirements.txt

EXPOSE 8787 
ENTRYPOINT ["/init"]