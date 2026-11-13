"""Versión realista de «Una tarde con papá».

Monta fotos fotorrealistas (video/fotos/*.jpg) con movimiento de cámara lento,
fundidos, subtítulos y voces naturales en español generadas con Piper TTS.

Requisitos: python3, pillow, numpy, ffmpeg (con rubberband), piper-tts y los
modelos de voz es_MX-ald-medium y es_MX-claude-high en VOCES_DIR.

Uso:  VOCES_DIR=/ruta/a/voces python3 video/generar_video_realista.py
"""
import math
import os
import subprocess
import sys
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 30
SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
FOTOS = os.path.join(HERE, "fotos")
OUT = os.path.join(HERE, "padre_e_hijo_realista.mp4")
VOCES = os.environ.get("VOCES_DIR", os.path.join(HERE, "voces"))
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_S = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

# Fotos: general, hijo, padre, conversacion, abrazo
# Cada toma: (foto, zoom_inicial, zoom_final, centro_inicial, centro_final) en coords. relativas
INTRO, OUTRO, FUNDIDO = 4.5, 5.0, 0.6

GUION = [
    # quien, texto, toma
    ("hijo", "Papá, ¿por qué el cielo se pone naranja en la tarde?", ("hijo", 1.0, 1.10, (0.5, 0.5), (0.45, 0.45))),
    ("padre", "Porque cuando el sol baja, su luz atraviesa mucho más aire, y solo los colores cálidos llegan hasta nosotros.", ("padre", 1.05, 1.15, (0.45, 0.5), (0.42, 0.45))),
    ("hijo", "¿Entonces el sol no cambia de color?", ("conversacion", 1.0, 1.08, (0.5, 0.5), (0.55, 0.48))),
    ("padre", "No, campeón. Somos nosotros quienes lo vemos distinto.", ("padre", 1.18, 1.25, (0.4, 0.42), (0.42, 0.45))),
    ("hijo", "Papá... ¿tú también tenías miedo cuando eras niño?", ("hijo", 1.15, 1.25, (0.42, 0.45), (0.4, 0.42))),
    ("padre", "Claro que sí. A la oscuridad, a los truenos... ¡y a los exámenes de matemáticas!", ("padre", 1.0, 1.08, (0.5, 0.5), (0.47, 0.47))),
    ("hijo", "¡Ja, ja! ¿Y cómo dejaste de tener miedo?", ("conversacion", 1.12, 1.2, (0.6, 0.5), (0.62, 0.47))),
    ("padre", "No dejé de tenerlo. Aprendí que ser valiente es tener miedo, y aun así intentarlo.", ("padre", 1.1, 1.22, (0.44, 0.47), (0.42, 0.42))),
    ("hijo", "Entonces mañana voy a levantar la mano en clase.", ("hijo", 1.0, 1.1, (0.48, 0.5), (0.43, 0.46))),
    ("padre", "Estoy muy orgulloso de ti. Y pase lo que pase, aquí voy a estar.", ("general", 1.2, 1.1, (0.38, 0.56), (0.4, 0.55))),
    ("hijo", "Te quiero, papá.", ("abrazo", 1.0, 1.06, (0.4, 0.55), (0.38, 0.55))),
    ("padre", "Y yo a ti, hijo. Vamos a casa, que se enfría la cena.", ("abrazo", 1.06, 1.14, (0.38, 0.55), (0.36, 0.53))),
]
TOMA_INTRO = ("general", 1.25, 1.12, (0.5, 0.55), (0.48, 0.55))
TOMA_OUTRO = ("general", 1.05, 1.0, (0.5, 0.5), (0.5, 0.5))

VOZ = {
    "padre": dict(modelo="es_MX-ald-medium", ritmo=1.08, filtro="rubberband=pitch=0.9:formant=preserved"),
    "hijo": dict(modelo="es_MX-claude-high", ritmo=0.95, filtro="rubberband=pitch=1.32:formant=shifted"),
}


# ---------------------------------------------------------------- audio
def leer_wav(ruta):
    with wave.open(ruta) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    assert sr == SR, (ruta, sr)
    return x


def sintetizar_voces(tmp):
    pistas = []
    for i, (quien, texto, _) in enumerate(GUION):
        v = VOZ[quien]
        cruda = os.path.join(tmp, f"l{i}_raw.wav")
        lista = os.path.join(tmp, f"l{i}.wav")
        subprocess.run([sys.executable, "-m", "piper", "-m", os.path.join(VOCES, v["modelo"] + ".onnx"),
                        "-f", cruda, "--length-scale", str(v["ritmo"]), "--sentence-silence", "0.35"],
                       input=texto.encode(), check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", cruda, "-af",
                        v["filtro"] + ",highpass=f=70,aecho=0.8:0.5:35:0.08",
                        "-ar", str(SR), "-ac", "1", lista], check=True)
        x = leer_wav(lista)
        # recorta silencios de los extremos
        nz = np.where(np.abs(x) > 0.01)[0]
        x = x[max(0, nz[0] - 800): nz[-1] + 1600]
        pistas.append(x / (np.abs(x).max() + 1e-9) * 0.85)
    return pistas


def linea_tiempo(pistas):
    t = INTRO
    lineas = []
    for (quien, texto, toma), x in zip(GUION, pistas):
        t += 0.45
        ini = t
        dur = len(x) / SR
        t += dur
        fin = t + 0.55
        lineas.append(dict(quien=quien, texto=texto, toma=toma, ini=ini, fin_voz=ini + dur, fin=fin, audio=x))
        t = fin
    total = t + OUTRO
    # tomas de video: cada línea cubre [ini - 0.45, fin)
    tomas = [(0.0, INTRO + 0.45, TOMA_INTRO)]
    for ln in lineas:
        a = tomas[-1][1]
        tomas.append((a, ln["fin"], ln["toma"]))
    tomas.append((tomas[-1][1], total, TOMA_OUTRO))
    return lineas, tomas, total


def mezclar_audio(lineas, total, ruta):
    n = int(total * SR) + SR
    t = np.arange(n) / SR
    voz = np.zeros(n)
    for ln in lineas:
        i0 = int(ln["ini"] * SR)
        voz[i0:i0 + len(ln["audio"])] += ln["audio"]

    rng = np.random.default_rng(1)
    # ambiente: viento suave (ruido rosa filtrado)
    ruido = np.cumsum(rng.normal(0, 1, n))
    ruido -= np.convolve(ruido, np.ones(2000) / 2000, mode="same")
    ruido /= np.abs(ruido).max()
    viento = ruido * (0.6 + 0.4 * np.sin(2 * np.pi * t / 9))
    # pájaros lejanos
    pajaros = np.zeros(n)
    tb = 1.0
    while tb < total - 2:
        for k in range(rng.integers(2, 5)):
            m = int(0.08 * SR)
            st = np.arange(m) / SR
            f = np.linspace(rng.uniform(3000, 4200), rng.uniform(2000, 2800), m)
            ch = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * st / 0.08) ** 2
            i0 = int((tb + k * 0.13) * SR)
            pajaros[i0:i0 + m] += ch
        tb += rng.uniform(3.5, 8)
    # música: piano suave (notas con armónicos y decaimiento) C - Am - F - G
    acordes = [[130.8, 196.0, 261.6, 329.6], [110.0, 164.8, 220.0, 261.6],
               [87.3, 130.8, 174.6, 220.0], [98.0, 146.8, 196.0, 246.9]]
    musica = np.zeros(n)
    paso = 0.75
    k = 0
    tt = 0.0
    while tt < total:
        ac = acordes[int(tt // 6) % 4]
        nota = ac[[0, 2, 1, 3, 2, 1, 3, 2][k % 8]] * (2 if k % 8 in (3, 6) else 1)
        m = min(int(3.0 * SR), n - int(tt * SR))
        st = np.arange(m) / SR
        tono = sum(a * np.sin(2 * np.pi * nota * h * st) for h, a in [(1, 1), (2, 0.35), (3, 0.12), (4, 0.05)])
        env = (1 - np.exp(-st * 80)) * np.exp(-st * 1.6)
        i0 = int(tt * SR)
        musica[i0:i0 + m] += tono * env * (0.9 if k % 8 == 0 else 0.6)
        if k % 8 == 0:  # bajo
            b = np.sin(2 * np.pi * ac[0] / 2 * st) * np.exp(-st * 0.8) * (1 - np.exp(-st * 40))
            musica[i0:i0 + m] += 0.8 * b
        k += 1
        tt += paso
    musica /= np.abs(musica).max()
    # baja la música mientras hablan
    hablando = np.convolve((np.abs(voz) > 0.02).astype(float), np.ones(SR // 2) / (SR // 2), mode="same")
    duck = 1 - 0.55 * np.clip(hablando * 4, 0, 1)
    fade = np.clip(np.minimum(t / 2.5, (total - t) / 3.5), 0, 1)
    pista = voz + fade * (0.16 * musica * duck + 0.03 * viento + 0.025 * pajaros)
    pista = np.tanh(pista * 1.05) * 0.95
    with wave.open(ruta, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((pista * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- video
def suave(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


IMGS = {}
F_SUB = F_TIT = F_PEQ = None
DATOS = None


def init_worker(datos):
    global DATOS, F_SUB, F_TIT, F_PEQ
    DATOS = datos
    for nombre in ("general", "hijo", "padre", "conversacion", "abrazo"):
        im = Image.open(os.path.join(FOTOS, nombre + ".jpg")).convert("RGB")
        # se escala para que cubra 1920x1080 con margen para el zoom
        k = max(W / im.width, H / im.height) * 1.0
        IMGS[nombre] = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS) if k > 1 else im
    F_SUB = ImageFont.truetype(FONT_R, 44)
    F_TIT = ImageFont.truetype(FONT_S, 96)
    F_PEQ = ImageFont.truetype(FONT_R, 34)


def encuadre(toma, k):
    nombre, z0, z1, c0, c1 = toma
    im = IMGS[nombre]
    e = suave(k) * 0.3 + k * 0.7
    z = max(1.0, z0 + (z1 - z0) * e)
    cx = c0[0] + (c1[0] - c0[0]) * e
    cy = c0[1] + (c1[1] - c0[1]) * e
    base = min(im.width / W, im.height / H)
    cw, ch = min(W * base / z, im.width), min(H * base / z, im.height)
    x0 = max(0.0, min(cx * im.width - cw / 2, im.width - cw))
    y0 = max(0.0, min(cy * im.height - ch / 2, im.height - ch))
    return im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


VIÑETA = None


def viñeta():
    global VIÑETA
    if VIÑETA is None:
        y, x = np.mgrid[0:H, 0:W]
        d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        VIÑETA = np.clip(1 - 0.35 * np.clip(d - 0.55, 0, None) ** 1.5, 0, 1)[..., None].astype(np.float32)
    return VIÑETA


def texto_sombra(d, pos, txt, fuente, color, sombra=6):
    x, y = pos
    for dx, dy in ((0, 3), (2, 2), (-2, 2), (0, 0)):
        pass
    d.text((x + 2, y + 3), txt, font=fuente, fill=(0, 0, 0, 200))
    d.text((x, y), txt, font=fuente, fill=color, stroke_width=2, stroke_fill=(0, 0, 0, 160))


def envolver(texto, fuente, ancho):
    out, act = [], ""
    for p in texto.split():
        prueba = (act + " " + p).strip()
        if fuente.getlength(prueba) <= ancho:
            act = prueba
        else:
            out.append(act)
            act = p
    return out + ([act] if act else [])


def frame(i):
    t = i / FPS
    lineas, tomas, total = DATOS
    # toma actual (y la siguiente para el fundido)
    img = None
    for j, (a, b, toma) in enumerate(tomas):
        if a <= t < b or j == len(tomas) - 1:
            k = (t - a) / max(b - a + FUNDIDO, 0.01)
            img = encuadre(toma, k)
            if j > 0 and t - a < FUNDIDO:
                pa, pb, ptoma = tomas[j - 1]
                prev = encuadre(ptoma, (t - pa) / max(pb - pa + FUNDIDO, 0.01))
                img = Image.blend(prev, img, suave((t - a) / FUNDIDO))
            break

    # gradación cálida + viñeta + grano suave
    arr = np.asarray(img).astype(np.float32)
    arr *= viñeta()
    arr[..., 0] *= 1.03
    arr[..., 2] *= 0.97
    rng = np.random.default_rng(i)
    arr += rng.normal(0, 3.0, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    # franjas de cine
    barra = 70
    d.rectangle([0, 0, W, barra], fill=(0, 0, 0, 255))
    d.rectangle([0, H - barra, W, H], fill=(0, 0, 0, 255))

    # subtítulos
    for ln in lineas:
        if ln["ini"] - 0.1 <= t < ln["fin"]:
            a = suave((t - ln["ini"] + 0.1) / 0.25) * suave((ln["fin"] - t) / 0.25)
            filas = envolver(ln["texto"], F_SUB, 1500)
            y = H - barra - 40 - len(filas) * 58
            nombre = "Papá" if ln["quien"] == "padre" else "Mateo"
            col_n = (255, 205, 120) if ln["quien"] == "padre" else (160, 210, 255)
            for r, fila in enumerate(filas):
                txt = (nombre + ":  " if r == 0 else "") + fila
                w = F_SUB.getlength(txt)
                x = (W - w) / 2
                al = int(255 * a)
                d.text((x + 2, y + 3 + r * 58), txt, font=F_SUB, fill=(0, 0, 0, int(180 * a)))
                if r == 0:
                    d.text((x, y), nombre + ":", font=F_SUB, fill=col_n + (al,),
                           stroke_width=2, stroke_fill=(0, 0, 0, int(140 * a)))
                    x += F_SUB.getlength(nombre + ":  ")
                    fila_txt = fila
                else:
                    fila_txt = fila
                d.text((x, y + r * 58), fila_txt, font=F_SUB, fill=(255, 255, 255, al),
                       stroke_width=2, stroke_fill=(0, 0, 0, int(140 * a)))
            break

    # título
    if t < INTRO + 0.3:
        a = suave((t - 0.8) / 1.0) * suave((INTRO + 0.3 - t) / 0.8)
        if a > 0:
            tit = "Una tarde con papá"
            w = F_TIT.getlength(tit)
            d.text(((W - w) / 2 + 3, 250 + 4), tit, font=F_TIT, fill=(0, 0, 0, int(150 * a)))
            d.text(((W - w) / 2, 250), tit, font=F_TIT, fill=(255, 236, 205, int(255 * a)))
            sub = "un cortometraje"
            w = F_PEQ.getlength(sub)
            d.text(((W - w) / 2, 380), sub, font=F_PEQ, fill=(255, 236, 205, int(220 * a)))

    img = Image.alpha_composite(img.convert("RGBA"), capa).convert("RGB")

    # fundidos de entrada/salida
    if t < 1.2:
        img = Image.blend(Image.new("RGB", (W, H)), img, suave(t / 1.2))
    if t > total - 3.5:
        k = suave((t - (total - 3.5)) / 1.8)
        img = Image.blend(img, Image.new("RGB", (W, H)), k)
        a = suave((t - (total - 2.6)) / 0.8) * suave((total - 0.3 - t) / 0.6)
        if a > 0:
            dd = ImageDraw.Draw(img)
            w = F_TIT.getlength("Fin")
            dd.text(((W - w) / 2, 470), "Fin", font=F_TIT, fill=tuple(int(c * a) for c in (255, 236, 205)))
    return img.tobytes()


def main():
    tmp = tempfile.mkdtemp()
    print("Generando voces…")
    pistas = sintetizar_voces(tmp)
    lineas, tomas, total = linea_tiempo(pistas)
    audio = os.path.join(tmp, "mezcla.wav")
    mezclar_audio(lineas, total, audio)
    datos = ([{k: v for k, v in ln.items() if k != "audio"} for ln in lineas], tomas, total)
    n = int(total * FPS)
    print(f"Duración {total:.1f} s, {n} cuadros")

    solo = [int(a) for a in sys.argv[1:]]
    if solo:
        init_worker(datos)
        for s in solo:
            Image.frombytes("RGB", (W, H), frame(s)).save(os.path.join(tmp, f"prev_{s}.png"))
        print("Vistas previas en", tmp)
        return

    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", audio,
        "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-tune", "grain", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-shortest", "-movflags", "+faststart", OUT,
    ], stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker, initargs=(datos,)) as pool:
        for k, data in enumerate(pool.imap(frame, range(n), chunksize=6)):
            ff.stdin.write(data)
            if k % 300 == 0:
                print(f"  {k}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("Listo:", OUT)


if __name__ == "__main__":
    main()
