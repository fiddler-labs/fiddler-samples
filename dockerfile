# based on the jupyter/scipy-notebook dockerfile
FROM jupyter/minimal-notebook:7a0c7325e470
COPY requirements.txt $HOME/
COPY jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

ENV PYTHONPATH .:/app/fiddler_samples

# Install Python 3 packages
RUN pip install -r requirements.txt
RUN rm -rf /opt/conda/lib/python3.7/site-packages/xgboost
RUN rm -rf /opt/conda/xgboost/libxgboost.so
ADD xgboost /opt/conda/lib/python3.7/site-packages/xgboost
ADD xgboost/libxgboost.so /opt/conda/xgboost

# Copy config to root's home directory. Enables running as root.
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# Start as root. Allows running jupyter as users other than jovyan.
USER root

WORKDIR /tmp

COPY ./runit.sh /app/runit.sh

CMD /app/runit.sh
