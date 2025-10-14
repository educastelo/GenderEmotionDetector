"""
Gender and Emotion Detection System

Este sistema realiza detecção facial em tempo real com classificação de gênero e emoção.
Utiliza três modelos YOLO:
- YOLOv11n-face: Para detecção de faces
- Gender Model: Para classificação de gênero (female/male)
- Emotion Model: Para classificação de emoções (5 categorias)

Autor: Eduardo
Data: 2025
"""

import cv2
import json
import os
import sys
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Definição das classes para classificação
CLASSES_GENDER = ['female', 'male']  # Classes de gênero suportadas
CLASSES_EMOTION = ['disgust', 'happy', 'neutral', 'surprise', 'unknown']  # Classes de emoção suportadas


def load_config(config_path="face_config.json"):
    """
    Carrega as configurações do sistema a partir de um arquivo JSON.
    
    O sistema suporta execução tanto como script Python quanto como executável compilado.
    Se o arquivo de configuração não for encontrado, usa valores padrão seguros.
    
    Args:
        config_path (str): Caminho relativo para o arquivo de configuração
        
    Returns:
        dict: Dicionário com as configurações do sistema contendo:
            - confidence_face: Threshold de confiança para detecção de faces
            - confidence_gender: Threshold de confiança para classificação de gênero
            - confidence_emotion: Threshold de confiança para classificação de emoção
            - resize_resolution: Resolução da janela de exibição
            - source_video: Fonte de vídeo (0=webcam, caminho=arquivo)
    """
    # Determinar o caminho base da aplicação (compatível com executáveis)
    if getattr(sys, 'frozen', False):
        # Executando como executável compilado (ex: PyInstaller)
        application_path = os.path.dirname(sys.executable)
    else:
        # Executando como script Python normal
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    config_file_path = os.path.join(application_path, config_path)
    
    try:
        # Tentar carregar configurações do arquivo JSON
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Configurações carregadas de: {config_file_path}")
        return config
    except FileNotFoundError:
        # Arquivo não encontrado - usar configurações padrão
        print(f"⚠️  Arquivo de configuração {config_file_path} não encontrado. Usando configurações padrão.")
        return {
            "confidence_face": 0.3,
            "confidence_gender": 0.3,
            "confidence_emotion": 0.3,
            "resize_resolution": "1280x720",
            "source_video": 0
        }
    except json.JSONDecodeError as e:
        # Arquivo JSON malformado - usar configurações padrão
        print(f"❌ Erro ao decodificar o arquivo {config_file_path}: {e}. Usando configurações padrão.")
        return {
            "confidence_face": 0.3,
            "confidence_gender": 0.3,
            "confidence_emotion": 0.3,
            "resize_resolution": "1280x720",
            "source_video": 0
        }

def corner_rect(img, bbox, l=30, t=5, rt=1,
                color_rect=(255, 0, 255), color_corners=(0, 255, 0)):
    """
    Draw a rectangle with highlighted corners.
    
    Args:
        img: Image to draw on
        bbox: Bounding box coordinates [x, y, w, h]
        l: Length of corner lines
        t: Thickness of corner lines
        rt: Thickness of main rectangle
        color_rect: Color of the main rectangle (BGR)
        color_corners: Color of the corner lines (BGR)
    
    Returns:
        Image with drawn rectangle and corners
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    
    # Draw main rectangle
    if rt != 0:
        cv2.rectangle(img, bbox, color_rect, rt)
    
    # Draw corners
    # Top Left
    cv2.line(img, (x, y), (x + l, y), color_corners, t)
    cv2.line(img, (x, y), (x, y + l), color_corners, t)
    # Top Right
    cv2.line(img, (x1, y), (x1 - l, y), color_corners, t)
    cv2.line(img, (x1, y), (x1, y + l), color_corners, t)
    # Bottom Left
    cv2.line(img, (x, y1), (x + l, y1), color_corners, t)
    cv2.line(img, (x, y1), (x, y1 - l), color_corners, t)
    # Bottom Right
    cv2.line(img, (x1, y1), (x1 - l, y1), color_corners, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), color_corners, t)
    
    return img

def put_text_rect(img, text, pos, scale=3, thickness=3,
                  color_text=(255, 255, 255), color_rect=(255, 0, 255),
                  font=cv2.FONT_HERSHEY_PLAIN, offset=10,
                  border=None, color_border=(0, 255, 0)):
    """
    Put text on image with rectangle background.
    
    Args:
        img: Image to draw on
        text: Text to display
        pos: Starting position (x, y)
        scale: Text scale
        thickness: Text thickness
        color_text: Text color (BGR)
        color_rect: Rectangle background color (BGR)
        font: OpenCV font type
        offset: Padding around text
        border: Border thickness (None for no border)
        color_border: Border color (BGR)
    
    Returns:
        tuple: (Modified image, Rectangle coordinates [x1, y2, x2, y1])
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    # Calculate rectangle coordinates
    x1, y1 = ox - offset, oy + offset
    x2, y2 = ox + w + offset, oy - h - offset
    
    # Draw background rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color_rect, cv2.FILLED)
    
    # Draw border if specified
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), color_border, border)
    
    # Draw text
    cv2.putText(img, text, (ox, oy), font, scale, color_text, thickness)
    
    return img, [x1, y2, x2, y1]

def load_models():
    """
    Carrega os três modelos YOLO necessários para o sistema:
    1. Modelo de detecção facial (YOLOv11n-face)
    2. Modelo de classificação de gênero
    3. Modelo de classificação de emoção
    
    Returns:
        tuple: (model_face, model_gender, model_emotion)
            - model_face: Modelo YOLO para detecção de faces
            - model_gender: Modelo YOLO para classificação de gênero
            - model_emotion: Modelo YOLO para classificação de emoção
            
    Raises:
        FileNotFoundError: Se algum dos arquivos de modelo não for encontrado
        Exception: Se houver erro ao carregar os modelos
    """
    # Determinar o caminho base para localizar os modelos
    if getattr(sys, 'frozen', False):
        # Executando como executável compilado
        application_path = os.path.dirname(sys.executable)
    else:
        # Executando como script Python
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    models_dir = os.path.join(application_path, "models")
    
    try:
        print("🔄 Carregando modelos YOLO...")
        
        # Carregar modelo de detecção facial
        face_model_path = os.path.join(models_dir, "yolov11n-face.pt")
        model_face = YOLO(face_model_path)
        print(f"✅ Modelo de detecção facial carregado: {face_model_path}")
        
        # Carregar modelo de classificação de gênero
        gender_model_path = os.path.join(models_dir, "gender.pt")
        model_gender = YOLO(gender_model_path)
        print(f"✅ Modelo de gênero carregado: {gender_model_path}")
        
        # Carregar modelo de classificação de emoção
        emotion_model_path = os.path.join(models_dir, "emotion.pt")
        model_emotion = YOLO(emotion_model_path)
        print(f"✅ Modelo de emoção carregado: {emotion_model_path}")
        
        return model_face, model_gender, model_emotion
        
    except FileNotFoundError as e:
        print(f"❌ Erro: Arquivo de modelo não encontrado - {e}")
        print("💡 Certifique-se de que todos os modelos estão na pasta 'models/':")
        print("   - yolov11n-face.pt")
        print("   - gender.pt")
        print("   - emotion.pt")
        raise
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        raise

def extract_face_region(frame, bbox):
    """
    Extrai a região da face detectada do frame original.
    
    Esta função é crucial para o pipeline de classificação, pois os modelos
    de gênero e emoção precisam da região da face isolada para fazer
    predições precisas.
    
    Args:
        frame (numpy.ndarray): Frame original da câmera/vídeo
        bbox (list/tuple): Coordenadas da bounding box [x1, y1, x2, y2]
        
    Returns:
        numpy.ndarray or None: Região da face extraída ou None se inválida
    """
    # Converter coordenadas para inteiros
    x1, y1, x2, y2 = map(int, bbox)
    
    # Obter dimensões do frame para validação
    h, w = frame.shape[:2]
    
    # Garantir que as coordenadas estão dentro dos limites válidos do frame
    # Isso previne erros de índice e garante extração segura
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Verificar se a bounding box é válida (tem área positiva)
    if x2 > x1 and y2 > y1:
        # Extrair e retornar a região da face
        face_region = frame[y1:y2, x1:x2]
        return face_region
    
    # Retornar None se a bounding box for inválida
    return None

def classify_face(model_gender, model_emotion, face_region, confidence_gender, confidence_emotion):
    """
    Realiza a classificação de gênero e emoção em uma região facial extraída.
    
    Esta função aplica os modelos de classificação na região da face e
    retorna apenas resultados que excedem os thresholds de confiança configurados.
    
    Args:
        model_gender (YOLO): Modelo treinado para classificação de gênero
        model_emotion (YOLO): Modelo treinado para classificação de emoção
        face_region (numpy.ndarray): Imagem da região facial extraída
        confidence_gender (float): Threshold mínimo de confiança para gênero
        confidence_emotion (float): Threshold mínimo de confiança para emoção
        
    Returns:
        tuple: (gender, emotion)
            - gender (str or None): Gênero classificado ou None se baixa confiança
            - emotion (str or None): Emoção classificada ou None se baixa confiança
    """
    # Validar entrada - região facial deve ser válida
    if face_region is None or face_region.size == 0:
        return None, None
    
    gender = None
    emotion = None
    
    # === CLASSIFICAÇÃO DE GÊNERO ===
    try:
        # Executar inferência do modelo de gênero
        gender_results = model_gender(face_region, verbose=False)
        
        if len(gender_results) > 0 and hasattr(gender_results[0], 'probs'):
            # Obter confiança da predição principal
            gender_confidence = gender_results[0].probs.top1conf.item()
            
            # Aceitar apenas predições com confiança suficiente
            if gender_confidence >= confidence_gender:
                gender_class_idx = gender_results[0].probs.top1
                gender = CLASSES_GENDER[gender_class_idx]
                
    except Exception as e:
        # Log silencioso - falhas na classificação não devem interromper o sistema
        pass
    
    # === CLASSIFICAÇÃO DE EMOÇÃO ===
    try:
        # Executar inferência do modelo de emoção
        emotion_results = model_emotion(face_region, verbose=False)
        
        if len(emotion_results) > 0 and hasattr(emotion_results[0], 'probs'):
            # Obter confiança da predição principal
            emotion_confidence = emotion_results[0].probs.top1conf.item()
            
            # Aceitar apenas predições com confiança suficiente
            if emotion_confidence >= confidence_emotion:
                emotion_class_idx = emotion_results[0].probs.top1
                emotion = CLASSES_EMOTION[emotion_class_idx]
                
    except Exception as e:
        # Log silencioso - falhas na classificação não devem interromper o sistema
        pass
    
    return gender, emotion

def detect_faces(model_face, model_gender, model_emotion, frame, confidence_face, confidence_gender, confidence_emotion):
    """
    Pipeline principal de detecção e classificação facial.
    
    Esta função coordena todo o processo:
    1. Detecta faces no frame usando YOLO
    2. Aplica blur para conformidade com LGPD
    3. Classifica gênero e emoção de cada face
    4. Desenha visualizações (bounding boxes e labels)
    
    Args:
        model_face (YOLO): Modelo para detecção de faces
        model_gender (YOLO): Modelo para classificação de gênero
        model_emotion (YOLO): Modelo para classificação de emoção
        frame (numpy.ndarray): Frame original da câmera/vídeo
        confidence_face (float): Threshold para detecção de faces
        confidence_gender (float): Threshold para classificação de gênero
        confidence_emotion (float): Threshold para classificação de emoção
        
    Returns:
        numpy.ndarray: Frame processado com detecções e anotações visuais
    """
    
    # === ETAPA 1: DETECÇÃO DE FACES ===
    # Executar detecção usando o modelo YOLO de faces
    face_results = model_face(frame, verbose=False, conf=confidence_face)
    face_detections = sv.Detections.from_ultralytics(face_results[0])
    
    # Criar cópia do frame para preservar o original
    frame_with_detections = frame.copy()
    
    # === ETAPA 2: APLICAÇÃO DE BLUR (CONFORMIDADE LGPD) ===
    # Aplicar blur automático em todas as faces detectadas para proteger privacidade
    if len(face_detections) > 0:
        blur_annotator = sv.BlurAnnotator(kernel_size=30)  # Blur forte para anonimização
        frame_with_detections = blur_annotator.annotate(
            scene=frame_with_detections, 
            detections=face_detections
        )
    
    # === ETAPA 3: CLASSIFICAÇÃO E ANOTAÇÃO ===
    # Processar cada face detectada individualmente
    for i, face_bbox in enumerate(face_detections.xyxy):
        
        # Extrair região da face do frame original (sem blur) para classificação
        face_region = extract_face_region(frame, face_bbox)
        
        # Classificar gênero e emoção da face extraída
        gender, emotion = classify_face(
            model_gender, model_emotion, face_region, 
            confidence_gender, confidence_emotion
        )
        
        # === CRIAÇÃO DO LABEL DINÂMICO ===
        # Construir label baseado nos resultados da classificação
        face_label = "Face"  # Label padrão
        
        if gender and emotion:
            # Ambas classificações disponíveis
            face_label = f"Face: {gender.title()} - {emotion.title()}"
        elif gender:
            # Apenas gênero disponível
            face_label = f"Face: {gender.title()}"
        elif emotion:
            # Apenas emoção disponível
            face_label = f"Face: {emotion.title()}"
        
        # === DESENHO DAS ANOTAÇÕES VISUAIS ===
        # Converter coordenadas da bounding box
        x1, y1, x2, y2 = map(int, face_bbox)
        w, h = x2 - x1, y2 - y1
        
        # Desenhar bounding box com cantos destacados (estilo futurista)
        corner_rect(
            frame_with_detections, 
            (x1, y1, w, h), 
            l=15,  # Comprimento das linhas dos cantos
            t=4,   # Espessura das linhas
            color_rect=(255, 0, 255),    # Magenta para o retângulo
            color_corners=(0, 255, 0)    # Verde para os cantos
        )
        
        # Desenhar label com fundo colorido
        put_text_rect(
            frame_with_detections,
            face_label,
            (max(0, x1), max(35, y1)),  # Posição ajustada para não sair da tela
            scale=0.8,
            thickness=2,
            color_rect=(224, 182, 90),   # Fundo azul acinzentado
            color_text=(40, 40, 40),     # Texto preto
            font=cv2.FONT_HERSHEY_DUPLEX,
            offset=5,
        )
    
    return frame_with_detections

def main():
    """
    Função principal do sistema de detecção de gênero e emoção.
    
    Coordena a inicialização dos modelos, captura de vídeo e loop principal
    de processamento em tempo real.
    """
    print("🚀 Iniciando Sistema de Detecção de Gênero e Emoção")
    print("="*50)
    
    try:
        # === INICIALIZAÇÃO ===
        # Carregar configurações do sistema
        config = load_config()
        
        # Configurar fonte de vídeo (webcam ou arquivo)
        source_video = config.get("source_video", 0)
        print(f"📹 Fonte de vídeo: {source_video}")
        
        # Inicializar captura de vídeo
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            raise Exception(f"Erro ao abrir fonte de vídeo: {source_video}")
            
        # Configurar resolução da câmera para melhor qualidade
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Carregar todos os modelos YOLO
        model_face, model_gender, model_emotion = load_models()
        
        print("\n✅ Sistema inicializado com sucesso!")
        print("💡 Pressione 'q' para sair")
        print("="*50)
        
        # === LOOP PRINCIPAL DE PROCESSAMENTO ===
        frame_count = 0
        
        while True:
            # Capturar frame da câmera/vídeo
            success, frame = cap.read()
            if not success:
                print("⚠️  Falha na captura do frame ou fim do vídeo")
                break
            
            frame_count += 1
            
            # Processar frame: detectar faces e classificar
            frame_with_detections = detect_faces(
                model_face,
                model_gender,
                model_emotion,
                frame,
                config["confidence_face"],
                config["confidence_gender"],
                config["confidence_emotion"]
            )
            
            # === REDIMENSIONAMENTO PARA EXIBIÇÃO ===
            resize_resolution = config["resize_resolution"]
            
            # Suportar múltiplos formatos de resolução
            if isinstance(resize_resolution, str) and 'x' in resize_resolution:
                # Formato "1280x720" - resolução fixa
                try:
                    width_str, height_str = resize_resolution.split('x')
                    resize_width = int(width_str)
                    resize_height = int(height_str)
                    resized = cv2.resize(frame_with_detections, (resize_width, resize_height))
                except ValueError:
                    # Fallback se formato for inválido
                    print(f"⚠️  Formato de resolução inválido: {resize_resolution}. Usando padrão.")
                    resized = cv2.resize(frame_with_detections, (1280, 720))
                    
            elif isinstance(resize_resolution, (int, float)):
                # Formato numérico - manter proporção baseada na largura
                resize_width = int(resize_resolution)
                original_height, original_width = frame_with_detections.shape[:2]
                resize_height = int(original_height * resize_width / original_width)
                resized = cv2.resize(frame_with_detections, (resize_width, resize_height))
                
            else:
                # Fallback para casos não previstos
                print(f"⚠️  Tipo de resolução não suportado: {type(resize_resolution)}. Usando padrão.")
                resized = cv2.resize(frame_with_detections, (1280, 720))
            
            # Exibir frame processado
            cv2.imshow("Gender & Emotion Detection System", resized)
            
            # === CONTROLE DE SAÍDA ===
            # Verificar se usuário pressionou 'q' para sair
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n👋 Encerrando sistema... (Processados {frame_count} frames)")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro no sistema: {e}")
    finally:
        # === LIMPEZA E FINALIZAÇÃO ===
        # Liberar recursos da câmera
        if 'cap' in locals():
            cap.release()
            
        # Fechar todas as janelas OpenCV
        cv2.destroyAllWindows()
        
        print("🔄 Recursos liberados. Sistema encerrado.")

if __name__ == "__main__":
    # Ponto de entrada do programa
    # Executa apenas se o arquivo for executado diretamente (não importado)
    main()
