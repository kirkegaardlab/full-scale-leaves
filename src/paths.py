from src import structure


class FilenameMaker:
    _formats = {'bool': '',
                'int': 'd',
                'float': '.3f',
                'float64': '.3f',
                'str': ''}

    def __init__(self, params):
        self._params = params
        self._im_name = self._params['im_name']
        self._body = self._make_body()
        self.pkl_filename = self._make_pkl_filename()

    @staticmethod
    def _get_type_str(val):
        type_ = type(val)
        type_str = type_.__name__
        return type_str

    def _make_body(self):
        bodyparts = []
        for name, val in self._params.items():
            # Do not include in filename
            if (name == 'optimize') or (name == 'use_gpu'):
                continue
            if name == 'im_name':
                val = self._im_name 
            type_str = self._get_type_str(val)
            format_ = self._formats[type_str]
            bodypart = name + '=' + format(val, format_)
            bodyparts.append(bodypart)

        body = '_'.join(bodyparts)

        return body

    def _make_pkl_filename(self):
        output_dir = (structure.Directories.model_output / self._im_name)
        filename = output_dir / (self._body + '.pkl')
        return filename

    def get_optimized_filename(self, dim):
        optimized_dir = structure.Directories.optimized
        n_edges_added = self._params['n_edges_added']
        gamma = self._params['gamma']

        if dim == '1d':
            gamma_str = f'_gamma={gamma:.3f}'
        else:
            gamma_str = ''
    
        filename = optimized_dir / (
            self._im_name 
            + f'_{dim}_n_edges_added={n_edges_added}{gamma_str}.pkl'
        )

        return filename

    def get_checkpoints_filename(self):
        checkpoints_dir = structure.Directories.checkpoints
        filename = checkpoints_dir / (self._body + '.txt')
        return filename

    def get_fig_filename(self, output_dir):
        filename = output_dir / (self._body + '.pdf')
        return filename
