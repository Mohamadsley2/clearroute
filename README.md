# ClearRoute 🗑️
Sistema de detecção de lixo nas ruas de Konstanz via vídeo + dashboard para o serviço de limpeza urbana.

---

## Estrutura de pastas

```
clearroute/
│
├── deteccao/          ← Script que vai rodar o modelo YOLO no vídeo
│                         (ainda não criado — próximo passo)
│
├── dashboard/         ← App Streamlit com mapa de calor e tabela de detecções
│                         (ainda não criado — próximo passo)
│
├── dados/             ← Arquivos JSON com as detecções (gerados pela detecção
│   └── dados_exemplo.json   ou usados como teste)
│
├── assets/            ← Imagens, logos ou outros arquivos estáticos
│
└── README.md          ← Este arquivo
```

---

## Campos do JSON de detecções

| Campo        | Tipo   | Descrição                                      |
|--------------|--------|------------------------------------------------|
| `lat`        | número | Latitude da detecção                           |
| `lon`        | número | Longitude da detecção                          |
| `typ`        | texto  | Tipo de lixo: Müll, Flasche, Verpackung, Becher |
| `konfidenz`  | número | Confiança do modelo (0 = 0%, 1 = 100%)         |
| `zeitstempel`| texto  | Momento do vídeo no formato MM:SS              |

---

## Próximos passos

1. `deteccao/detectar.py` — roda YOLO no vídeo e salva detecções em `dados/`
2. `dashboard/app.py` — lê o JSON e exibe mapa de calor + tabela no Streamlit

## Tecnologias
- Python 3.10+
- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [Streamlit](https://streamlit.io/)
- [Folium](https://python-visualization.github.io/folium/)
