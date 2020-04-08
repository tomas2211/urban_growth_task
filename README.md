# Urban growth task solution


Explanation of the approach and evaluation can be found in the report: [urban_growth_report.pdf](urban_growth_report.pdf).

## Setup

1. Clone the repo `git clone git@github.com:tomas2211/urban_growth_task.git`
2. Install requrements `pip install -r requirements.txt`
3. Download and unzip the dataset `./download_data.sh [link from task assignment]`

If you download the dataset elsewhere, specify the path by `--data_folder` parameter. The dataset folder **must** contain images in `imgs` folder and labels in `labels` folder.

## Usage

### Pre-trained models

Pre-trained models can be found in [models](models) folder.

### Visualizing urban index timeseries

To visualize the urban index timeseries and save the figures in visualizations folder, use the following command:

```shell script
python create_timeseries.py --device [cpu|cuda] --checkpoint_path models/[checkpoint] --out_folder visualizations
```

### Training

Training scripts with all parameter settings are located in [scripts](scripts_old) folder. Execute the script from the main directory.

### Model evaluation

Trained segmentation models can be evaluated by executing the following command:

```shell script
python eval_net.py --device [cpu|cuda] --checkpoint_path models/[checkpoint] --out_folder evaluation_figures
```

### Data analysis figures

If you want to enjoy the data analysis figures from the report as individual images, generate them using script `data_analysis.py`.