#!/bin/bash
tensorboard --port 6006 --logdir /tensorboard_summaries/ &
ipython notebook --ip='0.0.0.0' --NotebookApp.token='' --allow-root --port 8000 /notebooks/
