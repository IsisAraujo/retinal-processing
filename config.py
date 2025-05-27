from pathlib import Path

class HRFConfig:
    def __init__(self):
        # Parâmetros essenciais de processamento
        self.clahe_params = {
            'clip_limit': 2.0,
            'tile_grid': (8, 8),
            'extensions': ['.jpg', '.png']
        }

        # Diretórios base
        self.paths = {
            'input': Path('data/images'),
            'output': Path('processed'),
            'results': Path('results'),
            'visualizations': Path('visualizations')  # Novo diretório
        }

    def validate(self):
        """Validação inicial das configurações"""
        if not self.paths['input'].exists():
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {self.paths['input']}")

        if self.clahe_params['clip_limit'] <= 0:
            raise ValueError("Clip limit deve ser maior que zero")

        # Criar diretórios de saída se não existirem
        self.paths['visualizations'].mkdir(parents=True, exist_ok=True)
