import os, pathlib


class Directories:
    _src_path = os.path.abspath(os.path.dirname(__file__))
    project = pathlib.Path(_src_path).parent

    # Data
    data = project / 'data'
    images_originals = data / 'images_originals'
    images_upscaled_cropped = data / 'images_upscaled_cropped'
    images_bw = data / 'images_bw'
    probabilities = data / 'probabilities'
    extracted = data / 'extracted'
    reduced = data / 'reduced'
    added = data / 'added'
    data_paper = data / 'paper'

    # Model input/output
    model_output = data / 'model_output'
    checkpoints = model_output / 'checkpoints'
    optimized = model_output / 'optimized'

    # Scripts
    paper = project / 'scripts/paper'

    # Figures
    figures = project / 'figures'

    figures_extracted = figures / 'extracted'
    figures_reduced = figures / 'reduced'
    figures_added = figures / 'added'

    figures_artificial = figures / 'artificial'
    figures_chi2 = figures / 'chi2'
    figures_comparison = figures / 'comparison'
    figures_flows = figures / 'flows'
    figures_heatmap = figures / 'heatmap'
    figures_histogram = figures / 'histogram'
    figures_murray = figures / 'murray'
    figures_optimized = figures / 'optimized'
    figures_paper = figures / 'paper'

    @staticmethod
    def make_dir(path_name):
        path_name.mkdir(parents=True, exist_ok=True)
