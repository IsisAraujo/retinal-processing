from pathlib import Path

class HRFConfig:
    def __init__(self):
        # Parâmetros otimizados para imagens retinianas HRF
        self.clahe_params = {
            'clip_limit': 2.0,  # Conservador para preservar detalhes vasculares
            'tile_grid': (8, 8),  # Grid adequado para imagens de alta resolução
            'extensions': ['.jpg', '.png', '.tif', '.bmp', '.JPG']
        }

        # Configurações específicas do dataset HRF
        self.hrf_specs = {
            'expected_resolution': (3504, 2336),  # Resolução típica HRF
            'min_resolution': (1000, 1000),  # Mínimo aceitável
            'quality_threshold': 0.8  # Limiar de qualidade
        }

        # Diretórios
        self.paths = {
            'input': Path('data/images'),
            'output': Path('processed'),
            'results': Path('results'),
            'visualizations': Path('visualizations')
        }

    def validate(self):
        """Validação rigorosa das configurações"""
        if not self.paths['input'].exists():
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {self.paths['input']}")

        if self.clahe_params['clip_limit'] <= 0 or self.clahe_params['clip_limit'] > 10:
            raise ValueError("Clip limit deve estar entre 0 e 10")

        # Criar todos os diretórios necessários
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # Verificar se há imagens válidas
        images = list(self.paths['input'].glob('*'))
        if not images:
            raise ValueError(f"Nenhuma imagem encontrada em {self.paths['input']}")

        return True
