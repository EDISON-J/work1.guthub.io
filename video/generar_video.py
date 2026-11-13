"""Genera un video animado (MP4) de un padre y su hijo conversando al atardecer.

Todo se dibuja con Pillow y el audio (voces tipo "balbuceo", música y pájaros)
se sintetiza con numpy. Requiere: python3, pillow, numpy y ffmpeg.

Uso:  python3 video/generar_video.py  ->  video/padre_e_hijo.mp4
"""
import math
import os
import random
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
S = 2                      # supersampling para bordes suaves
FPS = 24
SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "padre_e_hijo.mp4")
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

INTRO = 3.5
OUTRO = 5.0

# ---------------------------------------------------------------- guion
# (personaje, texto, expresión)
GUION = [
    ("hijo", "Papá, ¿por qué el cielo se pone naranja en la tarde?", "curioso"),
    ("padre", "Porque cuando el sol baja, su luz atraviesa mucho más aire, y solo los colores cálidos llegan hasta nosotros.", "normal"),
    ("hijo", "¿Entonces el sol no cambia de color?", "curioso"),
    ("padre", "No, campeón. Somos nosotros quienes lo vemos distinto.", "feliz"),
    ("hijo", "Papá... ¿tú también tenías miedo cuando eras niño?", "triste"),
    ("padre", "Claro que sí. A la oscuridad, a los truenos... ¡y a los exámenes de matemáticas!", "feliz"),
    ("hijo", "¡Jajaja! ¿Y cómo dejaste de tener miedo?", "feliz"),
    ("padre", "No dejé de tenerlo. Aprendí que ser valiente es tener miedo y aun así intentarlo.", "normal"),
    ("hijo", "Entonces mañana voy a levantar la mano en clase.", "normal"),
    ("padre", "Estoy muy orgulloso de ti. Y pase lo que pase, aquí voy a estar.", "feliz"),
    ("hijo", "Te quiero, papá.", "feliz"),
    ("padre", "Y yo a ti, hijo. Vamos a casa, que se enfría la cena.", "feliz"),
]

VOZ = {
    "padre": dict(f0=118, sil=0.150, nombre="Papá", color=(52, 101, 164)),
    "hijo": dict(f0=265, sil=0.125, nombre="Mateo", color=(204, 62, 54)),
}

VOCALES = set("aeiouáéíóúü")


def silabas(palabra):
    n, prev = 0, False
    for c in palabra.lower():
        v = c in VOCALES
        if v and not prev:
            n += 1
        prev = v
    return max(1, n) if any(ch.isalpha() for ch in palabra) else 0


def construir_linea_tiempo():
    """Devuelve lista de líneas con sílabas programadas y la duración total."""
    rng = random.Random(7)
    t = INTRO
    lineas = []
    for quien, texto, expr in GUION:
        v = VOZ[quien]
        t += 0.35
        inicio = t
        sils = []           # (t0, dur, idx_fin_palabra, pitch_mult)
        pos = 0
        palabras = texto.split(" ")
        for wi, pal in enumerate(palabras):
            fin_pal = pos + len(pal)
            n = silabas(pal)
            for k in range(n):
                d = v["sil"] * rng.uniform(0.8, 1.25)
                pm = rng.uniform(0.88, 1.15)
                if k == n - 1 and pal.endswith("?"):
                    pm *= 1.3
                sils.append((t, d, fin_pal, pm))
                t += d
            if pal[-1:] in ",":
                t += 0.22
            elif pal.endswith("..."):
                t += 0.45
            elif pal[-1:] in ".!?" and wi < len(palabras) - 1:
                t += 0.35
            pos = fin_pal + 1
        t += 0.9            # tiempo para leer
        lineas.append(dict(quien=quien, texto=texto, expr=expr, ini=inicio,
                           fin=t, sils=sils))
    total = t + OUTRO
    return lineas, total


LINEAS, TOTAL = construir_linea_tiempo()


# ---------------------------------------------------------------- audio
def sintetizar_audio():
    n = int(TOTAL * SR) + SR
    pista = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(3)
    formantes = [(800, 1200), (500, 1900), (300, 2300), (500, 900), (350, 800)]

    for ln in LINEAS:
        v = VOZ[ln["quien"]]
        for (t0, d, _, pm) in ln["sils"]:
            m = int(d * SR)
            tt = np.arange(m) / SR
            f = v["f0"] * pm * (1 + 0.06 * np.sin(2 * np.pi * 5 * tt)) * np.linspace(1.04, 0.96, m)
            fase = 2 * np.pi * np.cumsum(f) / SR
            f1, f2 = formantes[rng.integers(len(formantes))]
            sig = np.zeros(m)
            for h in range(1, 30):
                fh = v["f0"] * pm * h
                if fh > 5000:
                    break
                amp = (np.exp(-((fh - f1) / 180) ** 2) + 0.6 * np.exp(-((fh - f2) / 250) ** 2) + 0.05) / h ** 0.4
                sig += amp * np.sin(h * fase)
            env = np.minimum(1, tt / 0.012) * np.minimum(1, (d - tt) / 0.04)
            sig = sig * np.clip(env, 0, 1)
            sig /= (np.abs(sig).max() + 1e-9)
            i0 = int(t0 * SR)
            pista[i0:i0 + m] += 0.32 * sig

    # música de fondo suave: C - Am - F - G
    t = np.arange(n) / SR
    acordes = [[261.6, 329.6, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 246.9, 293.7]]
    musica = np.zeros(n)
    dur_ac = 4.0
    for i in range(int(TOTAL / dur_ac) + 2):
        a0, a1 = int(i * dur_ac * SR), int((i + 1) * dur_ac * SR + 0.5 * SR)
        a1 = min(a1, n)
        if a0 >= n:
            break
        seg = t[a0:a1] - i * dur_ac
        env = np.minimum(1, seg / 0.8) * np.clip((dur_ac + 0.5 - seg) / 0.8, 0, 1)
        for fr in acordes[i % 4]:
            musica[a0:a1] += env * (np.sin(2 * np.pi * fr / 2 * seg) + 0.3 * np.sin(2 * np.pi * fr * seg))
        # arpegio delicado
        for j, fr in enumerate(acordes[i % 4] + [acordes[i % 4][0] * 2]):
            p0 = a0 + int(j * 1.0 * SR)
            m = min(int(1.2 * SR), n - p0)
            if m <= 0:
                continue
            st = np.arange(m) / SR
            musica[p0:p0 + m] += 0.5 * np.sin(2 * np.pi * fr * 2 * st) * np.exp(-st * 3)
    musica /= np.abs(musica).max()
    fade = np.clip(np.minimum(t / 2.0, (TOTAL - t) / 3.0), 0, 1)
    pista += 0.09 * musica * fade

    # pájaros
    rng2 = random.Random(11)
    tb = 1.0
    while tb < TOTAL - 3:
        for k in range(rng2.randint(2, 4)):
            m = int(0.07 * SR)
            st = np.arange(m) / SR
            f = np.linspace(rng2.uniform(3200, 4200), rng2.uniform(2200, 2800), m)
            ch = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * st / 0.07)
            i0 = int((tb + k * 0.11) * SR)
            pista[i0:i0 + m] += 0.04 * ch
        tb += rng2.uniform(4, 9)

    pista = np.tanh(pista * 1.1) * 0.9
    ruta = os.path.join(HERE, "_audio.wav")
    with wave.open(ruta, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((pista * 32767).astype(np.int16).tobytes())
    return ruta


# ---------------------------------------------------------------- dibujo
class Lienzo:
    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img)

    @staticmethod
    def s(v):
        return v * S

    def elipse(self, cx, cy, rx, ry, fill, outline=None, w=0):
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S],
                       fill=fill, outline=outline, width=int(w * S))

    def rect(self, x0, y0, x1, y1, fill, r=0, outline=None, w=0):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S,
                                 fill=fill, outline=outline, width=int(w * S))

    def poli(self, pts, fill, outline=None):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=outline)

    def linea(self, pts, fill, w):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=int(w * S), joint="curve")
        for x, y in (pts[0], pts[-1]):
            r = w / 2
            self.elipse(x, y, r, r, fill)

    def arco(self, cx, cy, rx, ry, a0, a1, fill, w):
        self.d.arc([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], a0, a1,
                   fill=fill, width=int(w * S))


def mezclar(c1, c2, k):
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


GROUND_Y = 600
SEAT_Y = 482


def fondo_estatico():
    img = Image.new("RGB", (W * S, H * S))
    L = Lienzo(img)
    # cielo degradado
    stops = [(0, (44, 42, 102)), (0.35, (122, 72, 140)), (0.62, (236, 120, 96)), (0.8, (252, 182, 104)), (1, (255, 214, 140))]
    horizonte = 505
    for y in range(horizonte * S):
        k = y / (horizonte * S)
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if a <= k <= b:
                c = mezclar(ca, cb, (k - a) / (b - a))
                break
        L.d.line([(0, y), (W * S, y)], fill=c)
    # estrellitas tempranas
    rng = random.Random(5)
    for _ in range(40):
        x, y = rng.uniform(0, W), rng.uniform(0, 140)
        L.elipse(x, y, 1.2, 1.2, (255, 240, 220))
    # sol con halo
    for r, c in [(130, (255, 200, 120)), (100, (255, 190, 110)), (75, (255, 170, 90)), (58, (255, 236, 170))]:
        L.elipse(640, 420, r, r, mezclar(c, (252, 182, 104), 0.5 if r > 60 else 0))
    # colinas lejanas
    L.poli([(0, 470)] + [(x, 430 - 35 * math.sin(x / 170) - 20 * math.sin(x / 63)) for x in range(0, W + 21, 20)] + [(W, 470)],
           (150, 86, 120))
    L.poli([(0, 500)] + [(x, 465 - 25 * math.sin(x / 120 + 2) - 10 * math.sin(x / 40)) for x in range(0, W + 21, 20)] + [(W, 500)],
           (112, 74, 108))
    # césped
    L.rect(0, 490, W, H, (88, 140, 72))
    for y in range(490, H, 4):
        L.rect(0, y, W, y + 4, mezclar((112, 160, 80), (54, 104, 58), (y - 490) / (H - 490)))
    # sendero
    L.poli([(560, 490), (720, 490), (980, H), (300, H)], (196, 160, 116))
    L.poli([(600, 490), (680, 490), (820, H), (460, H)], (210, 178, 132))
    # árboles
    for tx, ts in [(120, 1.15), (1160, 1.0), (1030, 0.6)]:
        L.rect(tx - 18 * ts, 300 * ts + 200 * (1 - ts) + 120, tx + 18 * ts, 520 + 20 * ts, (92, 60, 44))
        cy = 280 + 160 * (1 - ts) + 60
        for dx, dy, r in [(-60, 20, 75), (60, 20, 75), (0, -40, 90), (-30, 60, 70), (40, 60, 70)]:
            L.elipse(tx + dx * ts, cy + dy * ts, r * ts, r * ts, (60, 108, 62))
        for dx, dy, r in [(-40, -10, 50), (30, -60, 45), (60, 10, 40)]:
            L.elipse(tx + dx * ts, cy + dy * ts, r * ts, r * ts, (84, 136, 70))
    # flores
    rng = random.Random(9)
    for _ in range(60):
        x, y = rng.uniform(0, W), rng.uniform(520, H)
        if 300 < x < 980 and y < 700 and abs(x - 640) < (y - 490) * 0.8 + 90:
            continue
        c = rng.choice([(255, 220, 90), (255, 160, 190), (255, 255, 255), (200, 160, 255)])
        L.elipse(x, y, 4, 4, c)
        L.elipse(x, y, 1.6, 1.6, (255, 200, 60))
    # banca
    madera, madera2, hierro = (150, 94, 58), (120, 72, 44), (54, 54, 64)
    L.rect(392, SEAT_Y + 8, 402, 585, hierro)
    L.rect(878, SEAT_Y + 8, 888, 585, hierro)
    L.rect(385, 378, 395, SEAT_Y, hierro)
    L.rect(885, 378, 895, SEAT_Y, hierro)
    for y in (385, 415, 445):
        L.rect(372, y, 908, y + 22, madera, r=5)
        L.rect(372, y + 16, 908, y + 22, madera2, r=3)
    L.rect(366, SEAT_Y - 4, 914, SEAT_Y + 14, madera, r=6)
    L.rect(366, SEAT_Y + 8, 914, SEAT_Y + 14, madera2, r=3)
    # sombra
    L.elipse(640, 592, 300, 14, (66, 110, 56))
    return img


def nube(L, x, y, k, color):
    for dx, dy, r in [(0, 0, 30), (32, -12, 36), (68, 0, 30), (34, 8, 30)]:
        L.elipse(x + dx * k, y + dy * k, r * k, r * 0.8 * k, color)


def pajaro(L, x, y, fase):
    a = 8 * math.sin(fase)
    L.linea([(x - 12, y - a), (x, y), (x + 12, y - a)], (60, 40, 70), 2.2)


PIEL = (240, 196, 160)
PIEL_SOMBRA = (220, 170, 136)


def ojos(L, cx, cy, sep, r, mirar, parpadeo, expr):
    for sx in (-1, 1):
        ox = cx + sx * sep
        if expr == "feliz" and parpadeo < 0.5:
            L.arco(ox, cy + 3, r * 1.1, r, 200, 340, (40, 30, 30), 3)
            continue
        h = r * max(0.08, 1 - parpadeo)
        L.elipse(ox, cy, r * 0.95, h * 1.15, (255, 255, 255))
        if h > r * 0.3:
            L.elipse(ox + mirar * r * 0.35, cy + 1, r * 0.55, min(r * 0.62, h), (50, 36, 30))
            L.elipse(ox + mirar * r * 0.35 + 2, cy - 2, r * 0.18, r * 0.18, (255, 255, 255))
        else:
            L.linea([(ox - r, cy), (ox + r, cy)], (50, 36, 30), 2.5)


def boca(L, cx, cy, ancho, abierta, expr):
    if abierta > 0.08:
        h = 3 + abierta * ancho * 0.75
        L.elipse(cx, cy + h * 0.3, ancho * 0.55, h * 0.6, (110, 30, 40))
        L.elipse(cx, cy + h * 0.55, ancho * 0.32, h * 0.25, (230, 100, 110))
    else:
        if expr == "triste":
            L.arco(cx, cy + 10, ancho * 0.45, 8, 210, 330, (110, 40, 40), 3)
        else:
            L.arco(cx, cy - 6, ancho * 0.55, 10 if expr == "feliz" else 7, 20, 160, (110, 40, 40), 3)


def cejas(L, cx, cy, sep, expr, color, w):
    for sx in (-1, 1):
        x0 = cx + sx * sep
        if expr == "triste":
            pts = [(x0 - 10, cy + 2), (x0 + 10, cy - 4)] if sx < 0 else [(x0 - 10, cy - 4), (x0 + 10, cy + 2)]
        elif expr == "curioso":
            off = -5 if sx > 0 else 0
            pts = [(x0 - 10, cy + 2 + off), (x0, cy - 3 + off), (x0 + 10, cy + 1 + off)]
        else:
            pts = [(x0 - 10, cy + 2), (x0, cy - 2), (x0 + 10, cy + 2)]
        L.linea(pts, color, w)


def dibujar_padre(L, x, t, abierta, parpadeo, expr, mirar, gesto, abrazo):
    resp = 1.5 * math.sin(t * 1.8)
    bob = abierta * 2.5
    # piernas
    pant = (64, 70, 92)
    for dx in (-30, 30):
        L.rect(x + dx - 22, SEAT_Y - 10, x + dx + 22, 580, pant, r=10)
        L.rect(x + dx - 26, 570, x + dx + 30, 592, (60, 40, 32), r=10)
    # torso
    camisa = (68, 120, 180)
    ty = 322 + resp
    L.rect(x - 72, ty, x + 72, SEAT_Y + 12, camisa, r=40)
    L.poli([(x - 18, ty + 2), (x, ty + 30), (x + 18, ty + 2)], (236, 236, 240))
    L.poli([(x - 9, ty + 6), (x, ty + 26), (x + 9, ty + 6)], PIEL_SOMBRA)
    for k in range(3):
        L.elipse(x, ty + 50 + k * 30, 3, 3, (40, 80, 130))
    # brazo derecho (lado de la pantalla derecha) en la rodilla
    L.linea([(x + 64, ty + 30), (x + 82, ty + 100), (x + 56, SEAT_Y + 2)], camisa, 30)
    L.elipse(x + 50, SEAT_Y + 2, 15, 13, PIEL)
    # brazo izquierdo: gesto al hablar / abrazo
    g, a = gesto, abrazo
    def lerp(p, q, k):
        return (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)
    codo = lerp((x - 84, ty + 100 - 10 * g), (x - 120, ty + 78), a)
    mano = lerp((x - 60 - 50 * g, SEAT_Y + 2 - 95 * g), (x - 222, ty + 112), a)
    L.linea([(x - 64, ty + 30), codo, mano], camisa, 30)
    L.elipse(mano[0], mano[1], 15, 13, PIEL)
    # cabeza
    hx, hy = x + mirar * 4, 262 + resp + bob
    L.rect(hx - 16, hy + 40, hx + 16, ty + 14, PIEL_SOMBRA, r=8)
    L.elipse(hx - 56, hy + 6, 11, 15, PIEL_SOMBRA)
    L.elipse(hx + 56, hy + 6, 11, 15, PIEL_SOMBRA)
    L.elipse(hx, hy, 58, 64, PIEL)
    pelo = (74, 50, 36)
    L.elipse(hx, hy - 42, 56, 30, pelo)
    L.rect(hx - 58, hy - 48, hx - 44, hy, pelo, r=6)
    L.rect(hx + 44, hy - 48, hx + 58, hy, pelo, r=6)
    L.elipse(hx + 15, hy - 52, 36, 18, (90, 62, 44))
    # barba
    L.d.chord([(hx - 58) * S, (hy - 30) * S, (hx + 58) * S, (hy + 66) * S], 0, 180, fill=pelo)
    L.elipse(hx, hy + 28, 30, 20, PIEL)
    # ojos, cejas, nariz, boca
    ojos(L, hx + mirar * 3, hy - 6, 22, 9, mirar, parpadeo, expr)
    cejas(L, hx + mirar * 3, hy - 26, 22, expr, pelo, 5)
    L.elipse(hx + mirar * 5, hy + 12, 8, 7, PIEL_SOMBRA)
    L.rect(hx - 22 + mirar * 4, hy + 18, hx + 22 + mirar * 4, hy + 25, pelo, r=4)  # bigote
    boca(L, hx + mirar * 4, hy + 34, 26, abierta, expr)
    # lentes
    for sx in (-1, 1):
        L.d.rounded_rectangle([(hx + mirar * 3 + sx * 22 - 16) * S, (hy - 19) * S,
                               (hx + mirar * 3 + sx * 22 + 16) * S, (hy + 7) * S],
                              radius=8 * S, outline=(40, 40, 50), width=3 * S)
    L.linea([(hx + mirar * 3 - 6, hy - 8), (hx + mirar * 3 + 6, hy - 8)], (40, 40, 50), 3)


def dibujar_hijo(L, x, t, abierta, parpadeo, expr, mirar, apoyo):
    resp = 1.2 * math.sin(t * 2.3 + 1)
    bob = abierta * 2.5
    x = x + apoyo * 52
    inclin = apoyo * 10
    # piernas balanceándose (no llegan al suelo)
    for i, dx in enumerate((-18, 18)):
        sw = 10 * math.sin(t * 3.2 + i * math.pi) * (1 - apoyo)
        L.linea([(x + dx, SEAT_Y), (x + dx + sw * 0.4, SEAT_Y + 40), (x + dx + sw, SEAT_Y + 58)], (60, 92, 150), 24)
        L.elipse(x + dx + sw, SEAT_Y + 68, 15, 9, (240, 240, 240))
        L.rect(x + dx + sw - 15, SEAT_Y + 66, x + dx + sw + 15, SEAT_Y + 76, (210, 60, 60), r=4)
    # torso
    camiseta = (214, 76, 64)
    ty = 400 + resp
    L.rect(x - 48, ty, x + 48, SEAT_Y + 10, camiseta, r=28)
    L.elipse(x, ty + 46, 16, 16, (250, 210, 90))  # estrella/logo
    L.poli([(x, ty + 32), (x + 5, ty + 42), (x + 15, ty + 44), (x + 7, ty + 51), (x + 9, ty + 61),
            (x, ty + 55), (x - 9, ty + 61), (x - 7, ty + 51), (x - 15, ty + 44), (x - 5, ty + 42)], (255, 240, 200))
    # brazos
    L.linea([(x - 42, ty + 22), (x - 56, ty + 62), (x - 34, SEAT_Y + 4)], camiseta, 22)
    L.elipse(x - 32, SEAT_Y + 4, 11, 10, PIEL)
    L.linea([(x + 42, ty + 22), (x + 56, ty + 62), (x + 34, SEAT_Y + 4)], camiseta, 22)
    L.elipse(x + 32, SEAT_Y + 4, 11, 10, PIEL)
    # cabeza
    hx, hy = x + mirar * 3 + inclin, 352 + resp + bob + apoyo * 6
    L.rect(hx - 11, hy + 32, hx + 11, ty + 10, PIEL_SOMBRA, r=6)
    L.elipse(hx - 46, hy + 4, 9, 12, PIEL_SOMBRA)
    L.elipse(hx + 46, hy + 4, 9, 12, PIEL_SOMBRA)
    L.elipse(hx, hy, 48, 50, PIEL)
    pelo = (40, 28, 22)
    L.elipse(hx, hy - 30, 48, 24, pelo)
    for i in range(7):
        px = hx - 42 + i * 14
        L.poli([(px - 10, hy - 30), (px + 2, hy - 64 - (i % 2) * 8), (px + 12, hy - 30)], pelo)
    L.poli([(hx - 30, hy - 30), (hx - 8, hy - 14), (hx + 10, hy - 32)], pelo)  # flequillo
    # mejillas
    L.elipse(hx - 28, hy + 16, 9, 6, (246, 160, 150))
    L.elipse(hx + 28, hy + 16, 9, 6, (246, 160, 150))
    ojos(L, hx + mirar * 3, hy + 2, 17, 9.5, mirar, parpadeo, expr)
    cejas(L, hx + mirar * 3, hy - 16, 17, expr, pelo, 4)
    L.elipse(hx + mirar * 4, hy + 14, 5, 4, PIEL_SOMBRA)
    boca(L, hx + mirar * 4, hy + 28, 20, abierta, expr)


def corazon(L, x, y, r, color):
    L.elipse(x - r * 0.5, y, r * 0.55, r * 0.55, color)
    L.elipse(x + r * 0.5, y, r * 0.55, r * 0.55, color)
    L.poli([(x - r * 1.03, y + r * 0.15), (x + r * 1.03, y + r * 0.15), (x, y + r * 1.2)], color)


def envolver(texto, fuente, ancho):
    lineas, actual = [], ""
    for pal in texto.split(" "):
        prueba = (actual + " " + pal).strip()
        if fuente.getlength(prueba) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = pal
    if actual:
        lineas.append(actual)
    return lineas


F_TXT = ImageFont.truetype(FONT_R, 25 * S)
F_NOM = ImageFont.truetype(FONT_B, 18 * S)
F_TIT = ImageFont.truetype(FONT_B, 64 * S)
F_SUB = ImageFont.truetype(FONT_R, 26 * S)


def globo(L, quien, texto_total, visibles, alpha, cabeza):
    v = VOZ[quien]
    ancho_max = 470
    lineas = envolver(texto_total, F_TXT, ancho_max * S)
    alto_l = 34
    bw = max(F_TXT.getlength(l) for l in lineas) / S + 44
    bh = len(lineas) * alto_l + 56
    if quien == "hijo":
        bx, by = 70, 60
    else:
        bx, by = W - 70 - bw, 40
    escala = 0.85 + 0.15 * alpha
    cx, cy = bx + bw / 2, by + bh
    def tr(px, py):
        return (cx + (px - cx) * escala, cy + (py - cy) * escala)
    x0, y0 = tr(bx, by)
    x1, y1 = tr(bx + bw, by + bh)
    blanco = mezclar((255, 255, 255), (250, 200, 150), 1 - alpha)
    # cola
    base = min(max(cabeza[0], x0 + 40), x1 - 40)
    sombra = (90, 50, 70)
    L.poli([(base - 22, y1 - 4), (base + 22, y1 - 4), (cabeza[0], cabeza[1])], blanco)
    L.rect(x0 + 4, y0 + 6, x1 + 4, y1 + 6, mezclar(sombra, (240, 150, 110), 0.6), r=22)
    L.rect(x0, y0, x1, y1, blanco, r=22, outline=v["color"], w=3)
    L.poli([(base - 19, y1 - 5), (base + 19, y1 - 5), (cabeza[0] + 1, cabeza[1] - 6)], blanco)
    if alpha < 0.9:
        return
    L.d.text(((x0 + 22) * S, (y0 + 12) * S), v["nombre"], font=F_NOM, fill=v["color"])
    restante = visibles
    for i, l in enumerate(lineas):
        if restante <= 0:
            break
        trozo = l[:restante]
        restante -= len(l) + 1
        L.d.text(((x0 + 22) * S, (y0 + 40 + i * alto_l) * S), trozo, font=F_TXT, fill=(40, 34, 44))


def estado_habla(t):
    """Devuelve (linea_actual, apertura_boca, chars_visibles)."""
    for ln in LINEAS:
        if ln["ini"] - 0.3 <= t < ln["fin"]:
            abierta, vis = 0.0, 0
            for (t0, d, fin_pal, _) in ln["sils"]:
                if t >= t0:
                    vis = fin_pal
                if t0 <= t < t0 + d:
                    abierta = math.sin(math.pi * (t - t0) / d) ** 0.7
            return ln, abierta, vis
    return None, 0.0, 0


def parpadeo_en(t, semilla):
    rng = random.Random(semilla)
    tb = rng.uniform(1, 3)
    while tb < TOTAL:
        if tb <= t < tb + 0.16:
            return math.sin(math.pi * (t - tb) / 0.16)
        tb += rng.uniform(2.2, 5.0)
    return 0.0


def suave(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


FONDO = None


def init_worker():
    global FONDO
    FONDO = fondo_estatico()


FIN_DIALOGO = LINEAS[-1]["fin"]
T_ABRAZO = LINEAS[-2]["ini"]


def frame(i):
    t = i / FPS
    img = FONDO.copy()
    L = Lienzo(img)
    # nubes y pájaros
    for k, (y, vel, esc) in enumerate([(90, 6, 1.2), (160, 10, 0.9), (60, 4, 1.6), (210, 8, 0.7)]):
        x = (k * 380 + t * vel) % (W + 300) - 150
        nube(L, x, y, esc, mezclar((255, 200, 190), (255, 170, 150), k / 4))
    for k in range(3):
        bx = (t * 40 + k * 60) % (W + 600) - 300
        pajaro(L, bx, 140 + k * 18 + 6 * math.sin(t + k), t * 10 + k)

    ln, abierta, vis = estado_habla(t)
    quien = ln["quien"] if ln else None
    expr = ln["expr"] if ln else "feliz"
    # expresión: el que escucha también reacciona
    expr_p = expr if quien == "padre" else ("feliz" if expr == "feliz" else "normal")
    expr_h = expr if quien == "hijo" else ("feliz" if expr == "feliz" else "normal")
    if t > FIN_DIALOGO:
        expr_p = expr_h = "feliz"

    abrazo = suave((t - T_ABRAZO) / 1.2)
    gesto = 0.0
    if quien == "padre" and abrazo == 0:
        gesto = 0.5 * abierta + 0.25 * suave((t - ln["ini"]) / 0.5) * suave((ln["fin"] - 0.5 - t) / 0.5)

    xp, xh = 760, 520
    pb = parpadeo_en(t, 1)
    hb = parpadeo_en(t, 2)
    dibujar_hijo(L, xh, t, abierta if quien == "hijo" else 0, hb,
                 expr_h, 1.0, abrazo)
    dibujar_padre(L, xp, t, abierta if quien == "padre" else 0, pb, expr_p, -1.0, gesto, abrazo)

    # corazones al final
    if t > T_ABRAZO + 1.0:
        for k in range(5):
            tt = (t - T_ABRAZO - 1.0 - k * 0.7)
            if tt < 0:
                continue
            tt = tt % 3.5
            yy = 250 - tt * 45
            xx = 625 + 40 * math.sin(tt * 2 + k) + (k - 2) * 20
            corazon(L, xx, yy, 10 + 4 * math.sin(k), (232, 70, 100))

    # globo de diálogo
    if ln:
        alpha = suave((t - (ln["ini"] - 0.3)) / 0.25) * suave((ln["fin"] - t) / 0.2)
        cabeza = (500, 290) if quien == "hijo" else (790, 190)
        if quien == "hijo" and abrazo > 0:
            cabeza = (550, 290)
        globo(L, quien, ln["texto"], vis, alpha, cabeza)

    # títulos
    if t < INTRO + 0.5:
        a = suave(t / 0.8) * suave((INTRO + 0.3 - t) / 0.6)
        if a > 0.01:
            capa = img.copy()
            Lc = Lienzo(capa)
            Lc.rect(0, 250, W, 420, (30, 24, 50))
            tit, sub = "Una tarde con papá", "Una pequeña historia animada"
            wt = F_TIT.getlength(tit)
            Lc.d.text(((W * S - wt) / 2, 268 * S), tit, font=F_TIT, fill=(255, 226, 170))
            ws = F_SUB.getlength(sub)
            Lc.d.text(((W * S - ws) / 2, 360 * S), sub, font=F_SUB, fill=(240, 200, 210))
            img = Image.blend(img, capa, a * 0.9)

    out = img.resize((W, H), Image.LANCZOS)
    # fundidos
    if t < 0.8:
        out = Image.blend(Image.new("RGB", (W, H)), out, suave(t / 0.8))
    if t > TOTAL - 3.2:
        k = suave((t - (TOTAL - 3.2)) / 1.5)
        negro = Image.new("RGB", (W, H), (18, 12, 28))
        out = Image.blend(out, negro, k)
        if k > 0.5:
            d = ImageDraw.Draw(out)
            f = ImageFont.truetype(FONT_B, 56)
            a = suave((k - 0.5) * 2) * suave((TOTAL - 0.2 - t) / 0.8)
            col = mezclar((18, 12, 28), (255, 226, 170), a)
            d.text(((W - f.getlength("Fin")) / 2, 320), "Fin", font=f, fill=col)
    return out.tobytes()


def main():
    print(f"Duración: {TOTAL:.1f} s, {int(TOTAL * FPS)} cuadros")
    audio = sintetizar_audio()
    n = int(TOTAL * FPS)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", audio,
        "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", OUT,
    ], stdin=subprocess.PIPE)
    solo = [int(a) for a in sys.argv[1:]]
    if solo:   # modo vista previa: guarda PNGs de cuadros concretos
        init_worker()
        for s in solo:
            Image.frombytes("RGB", (W, H), frame(s)).save(os.path.join(HERE, f"_preview_{s}.png"))
        ff.stdin.close(); ff.kill()
        return
    with Pool(os.cpu_count(), initializer=init_worker) as pool:
        for k, data in enumerate(pool.imap(frame, range(n), chunksize=8)):
            ff.stdin.write(data)
            if k % 120 == 0:
                print(f"  {k}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(audio)
    print("Listo:", OUT)


if __name__ == "__main__":
    main()
