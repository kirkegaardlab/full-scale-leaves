import logging
import timeit

from src import model, optimizer, param_parser


# Set to '1d' or '2d'
_OPTIMIZER_DIM = '1d'


def _timer(func):
    def timed(*args, **kwargs):
        t_init = timeit.default_timer()
        res = func(*args, **kwargs)
        t_end = timeit.default_timer()

        print(f'Total time: {t_end - t_init:.4f} s')
        return res

    return timed


@_timer
def main():
    params = param_parser.get_params()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if params['optimize']:
        optimizer.run(params, _OPTIMIZER_DIM)
    else:
        model.run(params)


if __name__ == '__main__':
    main()
