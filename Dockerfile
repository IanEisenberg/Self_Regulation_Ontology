# Dockerfile for Self Regulation Ontology repo - Data Preperation

FROM python:3.5.3
MAINTAINER Russ Poldrack <poldrack@gmail.com>
RUN printf "deb http://archive.debian.org/debian/ jessie main\ndeb-src http://archive.debian.org/debian/ jessie main\ndeb http://security.debian.org jessie/updates main\ndeb-src http://security.debian.org jessie/updates main" > /etc/apt/sources.list
RUN apt-get update && apt-get install -y default-jre gfortran

# installing R
RUN wget https://cran.r-project.org/src/base/R-3/R-3.4.2.tar.gz
RUN tar zxf R-3.4.2.tar.gz
RUN cd R-3.4.2 && ./configure --enable-R-shlib=yes && make && make install

# installing R packages
RUN echo 'install.packages(c( \
  "doParallel", \
  "dplyr", \
  "dynamicTreeCut", \
  "foreach", \
  "iterators", \
  "glmnet", \
  "GPArotation", \
  "lme4", \
  "missForest", \
  "mpath", \
  "numDeriv", \
  "psych", \
  "pscl", \
  "qgraph", \
  "tidyr" \
  ), \
  repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && \
  Rscript /tmp/packages.R && \
  rm -rf /workdir/R-3.4.2*

# installing python packages
RUN pip install \
  cython==0.27.3 \ 
  git+https://github.com/IanEisenberg/dynamicTreeCut#eb822ebb32482a81519e32e944fd631fb9176b67 \
  imbalanced-learn==0.3.0 \
  ipdb \ 
  IPython==6.2.1 \
  Jinja2==2.9.6 \
  lmfit==0.9.7 \
  marshmallow==3.0.0b4 \
  matplotlib==2.1.0 \
  networkx==2.0 \
  nilearn==0.3.0 \
  numpy==1.11.1 \
  pandas==0.20.3 \
  python-igraph==0.7.1.post6 \
  requests==2.14.2 \
  scipy==0.19.1 \
  scikit-learn==0.19.0 \
  seaborn==0.7.1 \
  statsmodels==0.8.0 \
  svgutils==0.3.0 \
  jupyter

RUN pip install hdbscan==0.8.10 
# install hddm
RUN apt-get install -y liblapack-dev
RUN apt-get install -y libopenblas-dev
RUN pip install git+https://github.com/IanEisenberg/kabuki#4119a4c38fd7587109e86b5d12154df017903f7f
RUN pip install hddm==0.6.1

# set up rpy2
ENV C_INCLUDE_PATH /usr/local/lib/R/include
ENV LD_LIBRARY_PATH /usr/local/lib/R/lib
ENV IPYTHONDIR /tmp
# install more python packages that failed in first install
RUN pip install \
    git+https://github.com/IanEisenberg/expfactory-analysis#8316442e55ab6ce1031bc9f92148dae0283cfa9a \
    cvxpy==0.4.11 \
    fancyimpute==0.4.2 \
    joblib==0.11 \
    multiprocess==0.70.05 \
    rpy2==2.8.5
# packages just for analysis
RUN pip install bctpy==0.5.0

# Copy the directory (except Data and Results) into the docker container
ADD . /SRO
RUN mkdir /SRO/Data
RUN mkdir /SRO/Results
RUN mkdir /expfactory_token
RUN mkdir /Data
RUN mkdir /output
COPY example_data /Data/

# Create a settings file
RUN echo "expfactory_token:/expfactory_token/expfactory_token.txt" >> /SRO/Self_Regulation_Settings.txt
RUN echo "base_directory:/SRO" >> /SRO/Self_Regulation_Settings.txt
RUN echo "results_directory:/Results" >> /SRO/Self_Regulation_Settings.txt
RUN echo "data_directory:/Data" >> /SRO/Self_Regulation_Settings.txt

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

WORKDIR /SRO

# install the selfregulation directory 
RUN pip install -e /SRO

# Ensure user site-package directory isn't added to environment
RUN export PYTHONNOUSERSITE=1

CMD ["/bin/bash"]
