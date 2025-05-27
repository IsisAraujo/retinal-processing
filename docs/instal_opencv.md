# Compilação do OpenCV com Python 3.10 no Ubuntu usando pyenv

Este documento fornece instruções detalhadas para compilar o OpenCV com suporte ao Python 3.10 no Ubuntu, usando pyenv para gerenciar ambientes virtuais de Python.

## Pré-requisitos

1. **Instalar dependências do sistema:**

   ```bash
   sudo apt update
   sudo apt install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
   ```

2. **Configurar pyenv:**

   ```bash
   curl https://pyenv.run | bash
   echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Criar ambiente Python:**
   ```bash
   pyenv install 3.10.0
   pyenv virtualenv hrf-venv
   pyenv local hrf-venv
   ```

## Compilar e Instalar OpenCV

1. **Executar o script de build automatizado:**

   ```bash
   python build_opencv.py
   ```

   O script irá:

   - Instalar as dependências Python necessárias (numpy)
   - Clonar os repositórios do OpenCV
   - Configurar e compilar o OpenCV
   - Instalar no ambiente Python atual
   - Limpar arquivos temporários

## Verificar a instalação

1. **Testar OpenCV:**

   ```bash
   python -c 'import cv2; print(cv2.__version__)'
   ```

2. **Instalar dependências do projeto:**
   ```bash
   ./install_requirements.sh
   ```
