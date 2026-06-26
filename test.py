import cv2
from ultralytics import YOLO

print("1. Iniciando...")

# ---- Testa se o vídeo abre ----
caminho_video = r'C:\Users\AM\Desktop\VSCODe\clearroute\example3.mp4'
cap = cv2.VideoCapture(caminho_video)

if not cap.isOpened():
    print("ERRO: Não conseguiu abrir o vídeo. Verifica o caminho:")
    print(f"  {caminho_video}")
    exit()

print(f"2. Vídeo aberto. Total de frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

# ---- Extrai frame ----
cap.set(cv2.CAP_PROP_POS_FRAMES, 4)  # frame 4 (bem no início, mais seguro)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERRO: Conseguiu abrir o vídeo mas não leu o frame.")
    exit()

cv2.imwrite('frame_teste.jpg', frame)
print("3. Frame salvo como frame_teste.jpg")

# ---- Carrega o modelo ----
print("4. Carregando best.pt...")
modelo = YOLO('best.pt')
print(f"5. Modelo carregado. Classes: {list(modelo.names.values())}")

# ---- Roda detecção ----
print("6. Rodando detecção com conf=0.1...")
resultados = modelo.predict(
    source='frame_teste.jpg',
    conf=0.1,
    save=True,
    show=False,
    verbose=False
)

# ---- Resultado ----
deteccoes = resultados[0].boxes
if deteccoes is None or len(deteccoes) == 0:
    print("7. Nenhuma detecção mesmo com conf=0.1")
else:
    print(f"7. {len(deteccoes)} detecção(ões):")
    for box in deteccoes:
        classe = modelo.names[int(box.cls)]
        confianca = float(box.conf)
        print(f"   - {classe}: {confianca:.2f}")

print("8. Imagem anotada em: runs/detect/predict/frame_teste.jpg")
print("Feito.")