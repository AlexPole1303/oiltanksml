"""Generate the Cushing study plates from the existing project archive."""
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from contextlib import closing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager as fonts
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=HERE.parents[2])
parser.add_argument("--out", type=Path, default=HERE)
parser.add_argument("--fonts", type=Path, default=Path.home() / "Library/Fonts")
args = parser.parse_args()
ROOT, OUT = args.root.resolve(), args.out.resolve()
DERIVED = ROOT / "data/derived"
sys.path.insert(0, str(ROOT / "scripts"))
from build_cushing_holdout_market_features import numeric_prices
from build_weekly_shadow_panel import SOURCE, read_only
from fit_shadow_geometry import ideal_fields

def register_cmu(directory):
    for name in ("cmunrm", "cmunbx", "cmunti", "cmunbi"):
        original = TTFont(args.fonts / f"{name}.otf")
        order = original.getGlyphOrder()
        outlines = original.getGlyphSet()
        glyphs = {}
        for glyph in order:
            pen = TTGlyphPen(None)
            outlines[glyph].draw(Cu2QuPen(pen, .7, reverse_direction=True))
            glyphs[glyph] = pen.glyph()
        builder = FontBuilder(original["head"].unitsPerEm, isTTF=True)
        builder.setupGlyphOrder(order)
        builder.setupCharacterMap(original.getBestCmap())
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics(original["hmtx"].metrics)
        hhea = original["hhea"]
        builder.setupHorizontalHeader(ascent=hhea.ascent, descent=hhea.descent, lineGap=hhea.lineGap)
        os2 = original["OS/2"]
        if os2.usWeightClass == 500:
            os2.usWeightClass = 400
        builder.font["OS/2"] = os2
        builder.font["name"] = original["name"]
        builder.font["head"].macStyle = original["head"].macStyle
        builder.setupPost(italicAngle=original["post"].italicAngle)
        path = directory / f"{name}.ttf"
        builder.save(path)
        fonts.fontManager.addfont(path)


plt.rcParams.update({
    "font.family": "CMU Serif", "font.size": 11, "mathtext.fontset": "cm",
    "pdf.fonttype": 42, "axes.unicode_minus": False, "axes.edgecolor": "0.18",
    "axes.linewidth": .64, "xtick.color": "0.12", "ytick.color": "0.12",
    "hatch.linewidth": .35, "savefig.facecolor": "white",
})

NAMES = [
    ("01-ml-architecture", "Архитектура моделей машинного обучения"),
    ("02-shadow-physics", "Геометрия тени и изменение объёма"),
    ("03-ml-training", "Обучение и проверка моделей"),
    ("04-floating-roof", "Наземный резервуар с внешней плавающей крышей"),
    ("05-physical-model", "Прямая и обратная задачи оптической модели"),
    ("06-cushing-map", "Карта резервуаров нефтяного узла Кушинг"),
    ("07-eia-inventory", "Уровень и недельное изменение запасов Кушинга"),
    ("08-forecast-eia", "Предрелизный прогноз и первое значение EIA"),
    ("09-wti-prices", "Цены контрактов CL и календарный спред WTI"),
    ("10-strategy", "Накопленный результат стратегии на календарном спреде WTI"),
    ("11-ml-quality-detail", "Логистическая модель повторяемости измерения"),
    ("12-ml-registration", "Обучаемое совмещение снимков по неподвижным участкам"),
    ("13-terminal-consensus", "Согласованность спутниковых измерений по терминалам"),
    ("14-optical-revision", "Когда спутниковая поправка улучшает прогноз EIA"),
]


def write_text(ax, x, y, label, size=11, *, align="center", bold=False, italic=False, **kwargs):
    return ax.text(x, y, label, fontsize=size, ha=align, va="center",
                   weight="bold" if bold else "normal", style="italic" if italic else "normal",
                   linespacing=1.23, **kwargs)


def canvas(index, foot):
    f = plt.figure(figsize=(11, 7.15), facecolor="white")
    f.text(.5, .080, f"Рисунок {index}. {NAMES[index-1][1]}.", ha="center", va="center", fontsize=14)
    f.text(.5, .035, foot, va="center", ha="center", fontsize=8.8, color=".24", linespacing=1.25)
    return f


def drawing(f):
    ax = f.add_axes([.033, .139, .934, .82])
    ax.set(xlim=(0, 100), ylim=(0, 60))
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, title, body="", *, shade="white", size=11, body_size=10.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.75",
                               facecolor=shade, edgecolor="black", linewidth=.69, zorder=3))
    write_text(ax, x+w/2, y+h*(.73 if body else .50), title, size, bold=True, zorder=4)
    if body:
        write_text(ax, x+w/2, y+h*.32, body, body_size, zorder=4)


def line(ax, points, *, color="black", lw=.75, ls="-", z=2):
    xx, yy = zip(*points)
    ax.plot(xx, yy, lw=lw, ls=ls, color=color, zorder=z, solid_capstyle="round")


def arrow(ax, *points, dash=False, both=False, lw=.84, z=2):
    style = (0, (3.8, 3)) if dash else "-"
    if len(points) > 2:
        line(ax, points[:-1], lw=lw, ls=style, z=z)
    ax.add_patch(FancyArrowPatch(*points[-2:], arrowstyle="<->" if both else "-|>",
                                color="black", mutation_scale=10, shrinkA=0, shrinkB=0,
                                linewidth=lw, linestyle=style, zorder=z))


def group(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0,rounding_size=1",
                               fill=False, linewidth=.67, linestyle=(0,(5,4)), edgecolor=".2", zorder=1))
    write_text(ax, x+.65, y+h+1.45, title, 11.6, align="left", bold=True)


def lead(ax, x, y, label, target, *, align="left", size=10.2, elbow=None):
    write_text(ax, x, y, label, size, align=align)
    start = (x, y-1.25) if align == "center" else (x, y-1.2)
    arrow(ax, start, *([elbow] if elbow else []), target, lw=.57, z=5)


def architecture():
    f=canvas(1, "Сплошные стрелки показывают расчёт для выпуска. Пунктиром показаны инженерные проверки и обратная связь от EIA.")
    a=drawing(f)
    group(a,1,8,29,47,"Подготовка наблюдений")
    group(a,35,8,29,47,"Оценки на дату прогноза")
    group(a,69,8,30,47,"Прогноз и проверка")
    box(a,3,43,25,9,"Снимки и геометрия","Sentinel-2 B04 и Landsat B8\nортофото, реестр EFR, LiDAR")
    box(a,3,29,25,10,"Контуры и совмещение","Хаф + кольцевой контраст\nконтроль по неподвижным участкам")
    box(a,3,15,25,10,"Физическая инверсия","подгонка светотени для каждого бака\nоценка перепада $\\widehat g$ и диагностики",shade=".94")
    arrow(a,(15.5,43),(15.5,39)); arrow(a,(15.5,29),(15.5,25))
    box(a,37,43,25,9,"Публичный ансамбль","прошлые выпуски, региональный фон,\nсезонные и рыночные признаки",shade=".96")
    box(a,37,28,25,10,"Спутниковые признаки","пары оценок $\\widehat g$ → $\\widehat{\\Delta V}$\nсводка по площади, датам и охвату",shade=".96")
    box(a,37,13,25,10,"Инженерные модели","логистическая оценка качества\nробастная калибровка высоты")
    arrow(a,(28,20),(32.4,20),(32.4,33),(37,33))
    arrow(a,(28,17),(33.9,17),(33.9,18),(37,18),dash=True)
    arrow(a,(49.5,23),(49.5,28),dash=True)
    box(a,71,43,26,9,"Калибровка прогноза","история ошибок и текущий замер\nконтроль $b_t$ и поправка $u_t$")
    box(a,71,28,26,10,"Число до публикации","$\\widehat y_t=b_t+u_t$\nнедельное изменение по Кушингу",shade=".92")
    box(a,71,13,26,10,"Первое значение EIA","таблица 4 недельного отчёта WPSR\nцель $y_t$ и ошибка $y_t-\\widehat y_t$")
    arrow(a,(62,47.5),(71,47.5))
    arrow(a,(62,33),(66.6,33),(66.6,45),(71,45))
    arrow(a,(84,43),(84,38)); arrow(a,(84,28),(84,23))
    arrow(a,(84,13),(84,4),(49.5,4),(49.5,8),dash=True)
    write_text(a,65.5,2,"опубликованные исходы для следующих выпусков",10,italic=True)
    return f


def shadow_physics():
    f=canvas(2,"$g$ — перепад между кромкой и крышей, $L$ — длина тени вдоль проекции луча, $\\alpha$ — высота Солнца над горизонтом.\nУравнение справа предполагает свободное плавание крыши и постоянное сечение резервуара.")
    a=drawing(f)
    write_text(a,24,56,"(а) Сечение вдоль солнечного луча",12,bold=True)
    write_text(a,76,56,"(б) Из перепада высоты в баррели",12,bold=True)
    line(a,((52,7),(52,52)),ls=(0,(2,4)),color=".35")
    left,right,floor,top,roof = 11,45,14,44,29
    a.add_patch(Rectangle((left,floor),right-left,roof-floor,fc=".92",ec="none"))
    a.add_patch(Rectangle((left,roof),21,1.05,fc=".47",ec="none"))
    line(a,((left,top),(left,floor),(right,floor),(right,top)),lw=1.3)
    line(a,((left,roof),(right,roof)),lw=1.35)
    line(a,((left,top),(32,roof)),lw=1.0)
    arrow(a,(3,49.7),(left,top),(31.8,29.15),lw=.9)
    arrow(a,(13,51),(30,38.85),lw=.65)
    line(a,((left,top),(21,top)),color=".31",ls=(0,(3,3)))
    theta=np.linspace(0,-np.arctan2(top-roof,21),35)
    line(a,list(zip(left+6.5*np.cos(theta),top+6.5*np.sin(theta))),lw=.68)
    write_text(a,19.7,42,r"$\alpha$",14)
    write_text(a,32,49.7,"Луч Солнца",10.6,italic=True)
    arrow(a,(7.5,roof),(7.5,top),both=True)
    line(a,((6.7,roof),(left,roof)),lw=.52);line(a,((6.7,top),(left,top)),lw=.52)
    write_text(a,5.4,36.5,r"$g$",14)
    line(a,((left,roof),(left,24)),lw=.5);line(a,((32,roof),(32,24)),lw=.5)
    arrow(a,(left,24),(32,24),both=True)
    write_text(a,21.5,22,r"$L$",13)
    write_text(a,26,18.5,"Нефть",11,italic=True)
    lead(a,37.3,35.4,"Крыша",(38.2,29),align="center")
    write_text(a,28,8,r"$g=L\tan\alpha$",17)
    box(a,57,41,38,10,"Две даты одного резервуара","$g_{t_0}$ и $g_{t_1}$ от одной верхней кромки")
    box(a,57,25,38,11,"Изменение уровня и площадь","$\\widehat{\\Delta h}=\\widehat g_{t_0}-\\widehat g_{t_1}$,    $A=\\pi D^2/4$")
    box(a,57,9,38,11,"Изменение объёма","$\\widehat{\\Delta V}_{\\rm bbl}=A\\,\\widehat{\\Delta h}/0.158987294928$",shade=".92",body_size=13)
    arrow(a,(76,41),(76,36));arrow(a,(76,25),(76,20))
    return f


def training():
    f=canvas(3,"Хронология прогнозов воспроизводится по времени доступности исходных данных. Порог и параметры инженерного отбора\nпроверяются на группах по датам и терминалам. Сравнение прогнозов выполняется на одних и тех же выпусках.")
    a=drawing(f)
    group(a,1,16,37,40,"Инженерная ветвь")
    group(a,42,16,57,40,"Прогноз недельных запасов")
    box(a,3,42,33,11,"Примеры и метки","модуль разности оценок $\\leq 2$ м\nотдельный эталон высоты по LiDAR")
    box(a,3,29,33,10,"Подгонка параметров","11 диагностик → логит с L2 = 0,05\n6 признаков → калибровка soft-L1")
    box(a,3,18,33,8,"Проверка групп","исключение даты / целого терминала",shade=".93")
    arrow(a,(19.5,42),(19.5,39));arrow(a,(19.5,29),(19.5,26))
    box(a,44,42,25,11,"Снимок состояния на $t$","архив прошлых выпусков\nи доступные к отсечке снимки")
    box(a,73,42,24,11,"Цель для старых недель","из первого отчёта EIA\nс меткой времени публикации")
    box(a,51,29,39,10,"История с известным исходом","переоценка весов по прошлым ошибкам\nробастные потери и регуляризация",shade=".94")
    box(a,44,18,25,8,"Зафиксировать прогноз","до выхода отчёта $t$")
    box(a,73,18,24,8,"Сверить после выпуска","абсолютная ошибка и знак",shade=".94")
    arrow(a,(56.5,42),(56.5,39));arrow(a,(85,42),(85,39));arrow(a,(56.5,29),(56.5,26));arrow(a,(69,22),(73,22))
    arrow(a,(91,26),(91,34),(90,34),dash=True)
    line(a,((3,9),(97,9)),lw=.87)
    for x in (3,45,65,97):line(a,((x,8.1),(x,9.9)),lw=.75)
    for left,right,key,sub in [(3,45,"Разработка","2018–2022"),(45,65,"Валидация","2023"),(65,97,"Историческая проверка","2024–2026")]:
        a.add_patch(Rectangle((left+.25,9),right-left-.5,3.3,fc=".89",ec="none"))
        write_text(a,(left+right)/2,10.65,key,11,bold=True)
        write_text(a,(left+right)/2,6,sub,10.4)
    write_text(a,50,2,"Временные группы относятся к показанному сравнению прогнозов EIA.",9.6,italic=True)
    return f


def tank_section(a,x,y,h,*,mode):
    w,H=34,28
    a.add_patch(Rectangle((x,y),w,H,fc="white",ec="none",zorder=0))
    oil=h if mode!="landed" else h-3
    a.add_patch(Rectangle((x,y),w,oil,fc=".94",ec=".57",lw=.35,hatch="..",zorder=1))
    a.add_patch(Rectangle((x-1.3,y-1),w+2.6,1,fc=".88",ec=".15",lw=.55,hatch="///",zorder=2))
    line(a,((x,y+H),(x,y),(x+w,y),(x+w,y+H)),lw=1.45,z=5)
    line(a,((x+.38,y+H),(x+.38,y+.3),(x+w-.38,y+.3),(x+w-.38,y+H)),lw=.40,z=5)
    line(a,((x-.4,y+H),(x+.9,y+H)),lw=1.02,z=5)
    line(a,((x+w-.9,y+H),(x+w+.4,y+H)),lw=1.02,z=5)
    z=y+h
    line(a,((x+.8,z),(x+w-.8,z)),lw=1.08,z=5)
    for cx in (x+1.05,x+w-5.3):
        a.add_patch(Rectangle((cx,z-.55),4.25,2.25,fc="white",ec="black",lw=.80,zorder=5))
        line(a,((cx,z+.52),(cx+4.25,z+.52)),lw=.35,z=6)
    for cx in (x+.45,x+w-.95):
        a.add_patch(Rectangle((cx,z+.08),.53,.9,fc=".30",ec="black",lw=.44,hatch="////",zorder=6))
    for cx in (x+8,x+w-8):
        length = h if mode=="landed" else 4
        a.add_patch(Rectangle((cx-.20,z-length),.40,length+.95,fc="white",ec="black",lw=.54,zorder=6))
        line(a,((cx-.7,z-length),(cx+.7,z-length)),lw=.62,z=7)
    gx=x+w-7
    line(a,((gx,y+.35),(gx,y+H+.55),(x+w,y+H+.55)),lw=.52,z=7)
    a.add_patch(Rectangle((gx-.42,z-.42),.84,1.1,fc="white",ec="black",lw=.54,zorder=8))
    line(a,((x-3,y+1.85),(x+2.8,y+1.85)),lw=.58,z=6)
    line(a,((x-3,y+2.55),(x+2.8,y+2.55)),lw=.58,z=6)
    return z


def roof_tank():
    f=canvas(4,"Типовой понтонный резервуар EFR, разрезы вне масштаба. На посадочных опорах положение крыши\nперестаёт следовать за уровнем жидкости. Конструктивная схема по EPA AP-42, § 7.1.")
    a = drawing(f)
    write_text(a, 24, 58, "(а) Крыша свободно плавает", 13, bold=True)
    write_text(a, 75, 58, "(б) Крыша опирается на днище", 13, bold=True)
    tank_section(a, 7, 21, 15, mode="float")
    tank_section(a, 58, 21, 7, mode="landed")
    for x in (7, 58):
        for edge in (x, x + 34):
            line(a, ((edge, 18.7), (edge, 15.4)), lw=.55)
        arrow(a, (x, 17), (x + 34, 17), both=True, lw=.68)
        write_text(a, x + 17, 14.4, r"$D$ — внутренний диаметр", 11.5)
    arrow(a, (21.9, 39), (21.9, 46), both=True, lw=.77)

    def badge(number, x, y):
        a.scatter([x], [y], s=240, marker="o", facecolors="white",
                  edgecolors="black", linewidths=.7, zorder=10)
        write_text(a, x, y, str(number), 10.8, bold=True, zorder=11)

    callouts = [
        ("Обечайка", (3.2, 43), (7, 43)),
        ("Открытый верх", (22.5, 53.1), (22.5, 49.2)),
        ("Направляющая", (34.5, 53.1), (34, 49.55)),
        ("Понтон", (12.3, 41.9), (10.7, 37.7)),
        ("Кольцевое уплотнение", (38.25, 41.5), (40.35, 36.65)),
        ("Настил крыши", (27.5, 40), (27.5, 36)),
        ("Посадочная опора", (63.2, 34.5), (66, 28.8)),
        ("Крыша на опорах", (79.4, 36.5), (81, 28)),
        ("Уровень жидкости", (74.8, 26.5), (79.1, 25)),
        ("Патрубок", (51.3, 20.9), (55, 23.2)),
        ("Ход крыши", (18.5, 43), None),
    ]
    for number, (label, position, target) in enumerate(callouts, 1):
        if target is not None:
            a.add_patch(FancyArrowPatch(position, target, arrowstyle="-|>",
                                       color="black", mutation_scale=7.5,
                                       shrinkA=9.7, shrinkB=0, linewidth=.62, zorder=9))
        badge(number, *position)
        row, col = divmod(number - 1, 4)
        x, y = 2.5 + 25 * col, 10.1 - 3.6 * row
        badge(number, x, y)
        write_text(a, x + 2.05, y, label, 12.2, align="left", zorder=11)
    a.add_patch(Rectangle((76.5, 2.35), 2, 1.1, fc=".94", ec=".57", lw=.55, hatch=".."))
    write_text(a, 79.55, 2.9, "Нефть", 12.2, align="left")
    return f


def fields_image(ax, left, bottom, *, sun, view, gap, use_grid=False):
    grid=np.arange(-42,42.01,.5)
    masks=ideal_fields(grid[None,:],grid[:,None],30.,16.,np.array([gap]),sun,view)
    rim,ext,shadow=masks[:3]
    field=np.full(rim.shape,.97)
    field[rim]=.81
    field[ext]=.47
    field[shadow]=.31
    side=26
    ax.imshow(field, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              aspect="auto", extent=(left,left+side,bottom,bottom+side),zorder=1)
    ax.add_patch(Rectangle((left,bottom),side,side,fill=False,ec=".25",lw=.55,zorder=3))
    cx,cy=left+side/2,bottom+side/2
    radius=side*30/84
    offsets=gap*view*side/84
    units=np.linspace(0,2*np.pi,300)
    line(ax,zip(cx+radius*np.cos(units),cy+radius*np.sin(units)),lw=.72,z=4)
    line(ax,zip(cx+offsets[0]+radius*np.cos(units),cy+offsets[1]+radius*np.sin(units)),
         lw=.55,ls=(0,(2.5,2)),z=4)
    if use_grid:
        for t in np.arange(-40,41,10)*side/84:
            line(ax,((cx+t,bottom),(cx+t,bottom+side)),color=".35",lw=.36,z=3)
            line(ax,((left,cy+t),(left+side,cy+t)),color=".35",lw=.36,z=3)
    return cx,cy


def physical_model():
    f=canvas(5,"Геометрические маски слева рассчитаны кодом проекта для показательной конфигурации. Шаг 2 м относится к внутренней\nвычислительной сетке. Оценка опирается на исходные 10-метровые пиксели Sentinel-2 B04.")
    a=drawing(f)
    write_text(a,16,57,"(а) Проекция резервуара",11.8,bold=True)
    write_text(a,50,57,"(б) Прямая модель",11.8,bold=True)
    write_text(a,84,57,"(в) Обратная модель",11.8,bold=True)
    for x, y in ((33,5),(67,12)):
        line(a,((x,y),(x,54)),ls=(0,(2,3)),color=".48",lw=.58)
    sun,view=np.array((-1.12,.48)),np.array((.27,.38))
    fields_image(a,3,21,sun=sun,view=view,gap=9,use_grid=True)
    arrow(a,(11,48.5),(5.5,50.85),lw=.75)
    write_text(a,9.4,53.1,"к Солнцу",10)
    arrow(a,(20.5,48.5),(23.35,52.5),lw=.75)
    write_text(a,25,52.8,"обзор",10,align="left")
    for i,(c,caption) in enumerate([(".31","Тень на крыше / видимая стенка"),(".47","Тень за резервуаром"),(".81","Освещённая часть крыши")]):
        yy=16.3-i*3.1
        a.add_patch(Rectangle((3,yy-.74),2.0,1.48,fc=c,ec=".3",lw=.5))
        write_text(a,6,yy,caption,9.3,align="left")
    box(a,36,43,28,10,"Геометрия каждого $g$","контур, тени и параллакс\nпробный шаг 0,5 м",body_size=9.8)
    box(a,36,29,28,10,"Отклик прибора","свёртка с функцией рассеяния\nна расчётной сетке 2 м",body_size=9.8)
    box(a,36,15,28,10,"Сетка наблюдения","выборка модели на центрах\nисходных пикселей B04, 10 м",shade=".94",body_size=9.8)
    arrow(a,(50,43),(50,39));arrow(a,(50,29),(50,25))
    box(a,70,43,27,10,"Наблюдение $I_p$","отражение после масок облаков,\nснега и границ детектора",body_size=9.6)
    box(a,70,29,27,10,"Сравнение с моделью","сдвиг, фон, градиент и контраст\nтени могут только затемнять",body_size=9.7)
    box(a,70,13,27,12,"Допуск оценки $\\widehat g$","выигрыш над плоской крышей $\\geq 5$%\nминимум внутри диапазона,\nширина до 4 м, контроль Солнца",shade=".92",body_size=9.4)
    arrow(a,(83.5,43),(83.5,39));arrow(a,(83.5,29),(83.5,25))
    arrow(a,(64,20),(68.3,20),(68.3,34),(70,34))
    write_text(a,65.8,6.2,r"$\widehat g=\mathrm{arg\,min}_g\;\min_{\delta,\beta}\;\sum_p w_p\,[I_p-\widehat I_p(g,\delta,\beta)]^2$",12.3)
    line(a,((3,5.4),(5,5.4)),ls=(0,(2.5,2)),lw=.7)
    write_text(a,6,5.4,"Проекция контура крыши",9.3,align="left")
    return f


MAP_BOUNDS = (698600, 707500, 3976500, 3989700)
DETAIL_BOUNDS = (700800, 706200, 3978200, 3983000)
MAP_STYLE = {
    "weekly": dict(facecolor=".16", edgecolor="black", linewidth=.29, zorder=6),
    "other_efr": dict(facecolor="white", edgecolor=".07", linewidth=.52, zorder=5),
    "other_candidate": dict(facecolor=".87", edgecolor=".42", linewidth=.37, zorder=4),
}


def on_map(tanks, bounds):
    west,east,south,north=bounds
    return tanks.easting_m.between(west,east)&tanks.northing_m.between(south,north)


def map_inventory():
    registry=read("tank_engineering_registry").rename(columns={
        "footprint_east_m":"easting_m","footprint_north_m":"northing_m"})
    weekly=read("sentinel2_space_time_coverage_roof_registry")
    cols=["candidate_id","longitude","latitude","easting_m","northing_m","footprint_diameter_m",
          "documented_roof_type","documented_terminal_id"]
    tanks=registry[cols].copy()
    assert len(tanks)==454 and tanks.candidate_id.is_unique and weekly.candidate_id.is_unique
    assert weekly.has_both_dated_sources.all() and tanks.footprint_diameter_m.gt(0).all()
    is_weekly=tanks.candidate_id.isin(weekly.candidate_id)
    is_efr=tanks.documented_roof_type.eq("EFR")
    assert is_weekly.sum()==len(weekly) and (is_efr|~is_weekly).all()
    tanks["class"]=np.select([is_weekly,is_efr],["weekly","other_efr"],default="other_candidate")
    tanks["on_overview"]=on_map(tanks,MAP_BOUNDS)
    tanks["on_detail"]=on_map(tanks,DETAIL_BOUNDS)
    assert tanks.on_overview.sum()==435 and tanks.on_detail.sum()==318
    assert (tanks.on_overview&is_weekly).sum()==170
    return tanks


def kilometre_scale(ax, x, y, marks_km):
    for i,(left,right) in enumerate(zip(marks_km,marks_km[1:])):
        ax.add_patch(Rectangle((x+left*1000,y),1000*(right-left),65,
                  facecolor="black" if i%2==0 else "white",edgecolor="black",lw=.53,zorder=9))
    for n in marks_km:
        write_text(ax,x+n*1000,y-180,f"{n:g}".replace(".",",")+(f" км" if n==marks_km[-1] else ""),8.25,zorder=10)


def map_panel(ax, tanks, context, bounds):
    west,east,south,north=bounds
    ax.set(xlim=(west,east),ylim=(south,north),aspect="equal")
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_facecolor("white")
    for kind,style in (("street",dict(colors=".83",linewidths=.30,zorder=1)),
                       ("primary",dict(colors=".55",linewidths=.66,zorder=2)),
                       ("boundary",dict(colors=".35",linewidths=.68,linestyles=(0,(4,3)),zorder=3))):
        segments=[feature["coordinates_utm"] for feature in context["features"] if feature["type"]==kind]
        ax.add_collection(LineCollection(segments,**style))
    cropped=tanks[on_map(tanks,bounds)]
    for kind, style in MAP_STYLE.items():
        shapes=[Circle((t.easting_m,t.northing_m),t.footprint_diameter_m/2)
                for t in cropped[cropped["class"].eq(kind)].itertuples(index=False)]
        ax.add_collection(PatchCollection(shapes,**style))


def cushing_map(tanks):
    context=json.loads((HERE/"06-map-context.json").read_text())
    assert context["edition"]==2024 and context["crs"]=="EPSG:32614"
    f=canvas(6,"Круги построены в масштабе по центрам и диаметрам из реестра проекта. Вне обзорного кадра находятся 19 объектов реестра.\nДороги и граница города по TIGER/Line 2024. Проекция WGS 84 / UTM 14N.")
    overview=f.add_axes([.047,.169,.427,.704])
    detail=f.add_axes([.492,.342,.47,.531])
    map_panel(overview,tanks,context,MAP_BOUNDS)
    map_panel(detail,tanks,context,DETAIL_BOUNDS)
    overview.set_title("(а) Кушинг и нефтяной узел",fontsize=12.2,pad=9,weight="bold")
    detail.set_title("(б) Центральная группа резервуаров",fontsize=12.2,pad=9,weight="bold")
    x0,x1,y0,y1=DETAIL_BOUNDS
    overview.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fc="none",ec=".18",lw=.74,ls=(0,(2.5,2.5)),zorder=7))
    write_text(overview,706330,3983290,"(б)",9.4,zorder=10)
    cx,cy=context["city_reference_utm"]
    overview.plot(cx,cy,marker="o",markersize=2.9,color="black",zorder=8)
    write_text(overview,cx-230,cy+350,"Кушинг",13,align="right",bold=True,zorder=10,
               bbox=dict(fc="white",ec="none",pad=1.4,alpha=.93))
    write_text(overview,706660,3988610,"С",11.4,bold=True,zorder=10)
    arrow(overview,(706660,3987140),(706660,3988340),lw=.8,z=10)
    kilometre_scale(overview,699090,3977230,(0,1,2))
    kilometre_scale(detail,701110,3982610,(0,.5,1))
    f.text(.915,.318,"318 объектов на врезке",fontsize=9.6,ha="right")
    legend=f.add_axes([.511,.138,.44,.167])
    legend.set(xlim=(0,1),ylim=(0,1))
    legend.axis("off")
    counts=tanks[tanks.on_overview]["class"].value_counts()
    entries=[("weekly","Основная выборка измерений"),
             ("other_efr","Прочие с подтверждённой внешней крышей"),
             ("other_candidate","Прочие объекты реестра")]
    write_text(legend,.033,.92,"435 объектов на обзорной карте",11.4,align="left",bold=True)
    for i,(kind,text) in enumerate(entries):
        y=.72-.197*i
        style=MAP_STYLE[kind]
        legend.scatter(.047,y,s=27,facecolor=style["facecolor"],edgecolor=style["edgecolor"],lw=.6)
        write_text(legend,.085,y,text,10.0,align="left")
        write_text(legend,.966,y,str(counts[kind]),10.0,align="right")
    line(legend,((.026,.130),(.072,.130)),color=".35",ls=(0,(4,3)),lw=.7)
    write_text(legend,.085,.130,"Граница города",9.2,align="left")
    line(legend,((.50,.130),(.55,.130)),color=".57",lw=.75)
    write_text(legend,.565,.130,"Дороги",9.2,align="left")
    return f


def read(name, **kwargs):
    return pd.read_csv(DERIVED / f"{name}.csv", **kwargs)


def chart_axis(ax, *, ylim=None, major=None, year_gap=1):
    for s in ("right","top"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=".83", lw=.52, zorder=0)
    ax.tick_params(length=3.2,width=.57,labelsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(year_gap))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(major))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}".replace("-", "−").replace(".",",")))


def axes_pair(f, *, x=.102, w=.837, y1=.582, h1=.275, y2=.200, h2=.274):
    return f.add_axes([x,y1,w,h1]),f.add_axes([x,y2,w,h2])


def largest_eia():
    history=read("eia_inventory_release_panel")
    recent=read("eia_wpsr_forward_releases").rename(columns={
        "latest_week_in_issue":"week_ending", "cushing_change_kbbl":"change_kbbl_as_known_at_release"})
    cols=["week_ending","release_date","cushing_kbbl","change_kbbl_as_known_at_release","source_url"]
    panel=pd.concat((history[cols],recent[cols]),ignore_index=True).sort_values("week_ending")
    if panel.week_ending.duplicated().any() or panel[cols].isna().any().any():
        raise ValueError("Invalid EIA panel")
    panel["date"]=pd.to_datetime(panel["week_ending"])
    return panel


def eia_inventory(panel):
    f=canvas(7,"Первые опубликованные значения недельного WPSR, таблица 4. Изменение вычислено по значениям\nтекущей и предыдущей недели в том же выпуске. Ось времени показывает конец отчётной недели.")
    a,b=axes_pair(f,y1=.593,h1=.286,y2=.223,h2=.246)
    a.set_title("(а) Уровень коммерческих запасов", loc="left", fontsize=12, pad=8)
    b.set_title("(б) Изменение запасов к предыдущей неделе", loc="left", fontsize=12, pad=8)
    chart_axis(a,major=10)
    chart_axis(b,major=2)
    dates=panel.date
    for ax in (a,b):
        ax.set_xlim(dates.min()-pd.Timedelta(days=45),dates.max()+pd.Timedelta(days=95))
        ax.set_ylabel("млн баррелей",fontsize=10,labelpad=8)
    a.plot(dates,panel.cushing_kbbl/1000,color=".05",lw=1.1)
    b.axhline(0,color=".27",lw=.62,zorder=1)
    delta=panel.change_kbbl_as_known_at_release/1000
    b.vlines(dates,0,delta,colors=".22",lw=.60,zorder=2)
    last=panel.iloc[-1]
    a.scatter(last.date,last.cushing_kbbl/1000,s=11,c="black",zorder=5)
    a.annotate(f"{last.cushing_kbbl/1000:.3f}".replace(".",","),(last.date,last.cushing_kbbl/1000),
               xytext=(-12,19),textcoords="offset points",ha="right",fontsize=10,
               arrowprops=dict(arrowstyle="-",color=".2",lw=.55))
    f.text(.94,.928,f"{len(panel)} отчётные недели  |  {dates.min():%d.%m.%Y}–{dates.max():%d.%m.%Y}",
           ha="right",fontsize=9.6,color=".23")
    return f


def forecast_values(eia):
    forecast=read("cushing_optical_direction_forecasts")
    q=forecast.query('channel=="inventory" and study_phase=="holdout"').sort_values("release_date").copy()
    ledger=q.merge(eia[["week_ending","release_date","change_kbbl_as_known_at_release"]],on=["week_ending","release_date"],
                   validate="one_to_one")
    assert len(ledger)==len(q)==148
    assert np.allclose(1000*ledger.actual,ledger.change_kbbl_as_known_at_release)
    clocks=read("cushing_holdout_eia_vintage_audit_releases")
    clock=q.merge(clocks[["release_date","scheduled_release_at_utc","timing_basis"]],on="release_date",validate="one_to_one")
    before, scheduled, after=(pd.to_datetime(clock[c],utc=True,format="mixed")
                              for c in ("cutoff","scheduled_release_at_utc","known"))
    exact=scheduled.notna()
    assert len(clock)==len(q) and (before[exact].lt(scheduled[exact])&scheduled[exact].lt(after[exact])).all()
    new_york="America/New_York"
    assert clock.loc[~exact,"timing_basis"].eq("archived_day_only").all()
    start=before[~exact].dt.tz_convert(new_york)
    end=after[~exact].dt.tz_convert(new_york)
    assert start.dt.strftime("%Y-%m-%d").eq(clock.loc[~exact,"release_date"]).all()
    assert start.eq(start.dt.normalize()).all() and end.eq(start+pd.DateOffset(days=1)).all()
    q["date"]=pd.to_datetime(q["release_date"])
    info=json.loads((DERIVED/"cushing_optical_direction_summary.json").read_text())["phases"]["holdout"]["inventory"]["all"]
    for m in ("no_revision","same_week","reversed_direction"):
        assert np.isclose((1000*(q[m]-q.actual).abs()).mean(), info[m]["mae_kbbl"])
    return q,info


def forecast_eia(q,info):
    f=canvas(8,"Парные различия MAE рассчитаны по одним и тем же 148 выпускам. 95%-е интервалы получены блочным бутстрэпом.\nИстория уже использовалась в исследованиях проекта.")
    a=f.add_axes([.104,.527,.828,.350])
    b=f.add_axes([.289,.203,.612,.183])
    a.set_title("(а) Недельное изменение перед выходом отчёта и после публикации",loc="left",fontsize=12,pad=12)
    chart_axis(a,major=2)
    a.set_xlim(pd.Timestamp("2023-12-23"),pd.Timestamp("2026-12-01"))
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    a.set_ylabel("млн баррелей",fontsize=10,labelpad=7)
    a.axhline(0,color=".52",lw=.55)
    a.plot(q.date,q.actual,c=".12",lw=.92,label="Первая публикация EIA",zorder=3)
    a.plot(q.date,q.same_week,c=".30",ls=(0,(4,2)),lw=1.0,label="Прогноз со спутниковой поправкой",zorder=4)
    a.legend(frameon=False,ncol=2,fontsize=9.6,loc="upper left",bbox_to_anchor=(0,1.025))
    control=info["no_revision"]; same=info["same_week"]; reverse=info["reversed_direction"]
    b.set_title("(б) Разница MAE относительно прогноза с текущей оптической поправкой",loc="left",fontsize=11.8,pad=9)
    b.spines[["top","right","left"]].set_visible(False)
    b.grid(axis="x",color=".85",lw=.55)
    b.axvline(0,color=".24",lw=.7,ls=(0,(2,3)))
    b.set_xlim(-25,115)
    b.set_ylim(-.48,1.5)
    b.set_yticks([1,0],[f"Поправка исключена ({control['mae_kbbl']:.2f})".replace(".",","),
                     f"Знак обращён ({reverse['mae_kbbl']:.2f})".replace(".",",")])
    b.tick_params(axis="y",length=0,pad=8,labelsize=10.4)
    b.tick_params(axis="x",length=2.6,labelsize=10)
    b.set_xlabel("разница MAE, тыс. баррелей",labelpad=3,fontsize=10)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:.0f}".replace("-","−")))
    for y,d in ((1,control),(0,reverse)):
        val=d["mae_kbbl"]-same["mae_kbbl"]
        lo,hi=d["delta_vs_same_week_95_kbbl"]
        b.errorbar(val,y,xerr=np.array([[val-lo],[hi-val]]),fmt="o",ms=4.1,color=".05",
                   capsize=3,elinewidth=.85,lw=.75,zorder=5)
    f.text(.934,.428,f"MAE со спутниковой поправкой  {same['mae_kbbl']:.2f} тыс. баррелей".replace(".",","),
           ha="right",fontsize=9.8,color=".17")
    return f


def market_ledger():
    w=read("cushing_cl_extension_segments_trades").query('method=="curve_post_report_gate"').copy()
    v=read("cushing_cl_delayed_calendar_trades").query('method=="public"')
    w=w.merge(v[["release_date","near_month","far_month","source"]],on=["release_date","near_month","far_month"],
              validate="one_to_one")
    assert len(w)==389 and w.release_date.is_unique
    marks=read("wti_cushing_event_bars",dtype={"delivery_month":str})
    marks=marks[marks.session_number.ge(2)&marks.release_date.isin(w.release_date)]
    mark_book={(r.release_date,r.session_number,r.leg):r for r in marks.itertuples(index=False)}
    prices=[]
    with closing(read_only(SOURCE)) as db:
        value=numeric_prices(db, {"archive","ibkr"}, through=max(marks.session_date))
        for r in w.sort_values("release_date").itertuples(index=False):
            row={"release_date":r.release_date,"study_phase":r.study_phase,"source":r.source,
                 "side":r.near_side,"gross_usd":r.gross_usd,"near_month":r.near_month,"far_month":r.far_month,
                 "entry_at_new_york":r.entry_at_utc,"exit_at_new_york":r.exit_at_utc}
            for n,where,clock in ((2,"entry",r.entry_at_utc),(r.exit_session_number,"exit",r.exit_at_utc)):
                for leg in ("near","far"):
                    k=mark_book[r.release_date,n,leg]
                    assert k.delivery_month==str(getattr(r,f"{leg}_month")) and k.settlement_reference_time==clock
                    row[f"{where}_{leg}_usd_bbl"]=value(r.source,k.session_date,k.delivery_month,getattr(k,f"{r.source}_reference"))
                row[f"{where}_spread_usd_bbl"]=round(row[f"{where}_near_usd_bbl"]-row[f"{where}_far_usd_bbl"],2)
            pnl=round(1000*r.near_side*(row["exit_spread_usd_bbl"]-row["entry_spread_usd_bbl"]))
            assert pnl==r.gross_usd, (r.release_date,pnl,r.gross_usd)
            prices.append(row)
    p=pd.DataFrame(prices)
    p["entry_date"]=pd.to_datetime(p["entry_at_new_york"].str[:10])
    p["exit_date"]=pd.to_datetime(p["exit_at_new_york"].str[:10])
    t=p[p.side.ne(0)].sort_values("exit_at_new_york").copy()
    t["calendar_year"]=t.release_date.str[:4]
    t["cumulative_gross_usd"]=t.gross_usd.cumsum()
    assert t.gross_usd.sum()==73980 and t[t.study_phase.eq("holdout")].gross_usd.sum()==62050 and len(t)==70
    return p,t


def set_market_axis(ax,p,major):
    chart_axis(ax,major=major)
    ax.set_xlim(p.entry_date.min()-pd.Timedelta(days=35),p.exit_date.max()+pd.Timedelta(days=65))
    ax.set_ylabel("долл. США за баррель",fontsize=10,labelpad=7)


def wti_prices(p):
    f=canvas(9,"Каждая пара точек соответствует расчётной отметке второй полной сессии после выпуска EIA. Месяцы поставки\nразличаются на три месяца и фиксируются внутри окна. Закрытия взяты из архивов TradingCharts и IBKR.")
    a,b=axes_pair(f,y1=.586,h1=.286,y2=.216,h2=.259)
    a.set_title("(а) Цены ближнего и дальнего контрактов CL",loc="left",fontsize=12,pad=8)
    b.set_title("(б) Календарный спред (ближний минус дальний)",loc="left",fontsize=12,pad=8)
    set_market_axis(a,p,20);set_market_axis(b,p,4)
    a.scatter(p.entry_date,p.entry_near_usd_bbl,s=7.8,c=".12",lw=0,label="Ближний CL",zorder=4)
    a.scatter(p.entry_date,p.entry_far_usd_bbl,s=13.0,fc="white",ec=".38",lw=.6,label="Дальний CL",zorder=3)
    a.legend(frameon=False,fontsize=10,ncol=2,loc="lower right",bbox_to_anchor=(1,1.01))
    b.axhline(0,color=".24",lw=.62)
    b.vlines(p.entry_date,0,p.entry_spread_usd_bbl,color=".42",lw=.46)
    b.scatter(p.entry_date,p.entry_spread_usd_bbl,s=5.6,c=".14",lw=0,zorder=4)
    f.text(.938,.926,f"{len(p)} окон  |  {p.entry_date.min():%d.%m.%Y}–{p.entry_date.max():%d.%m.%Y}",ha="right",fontsize=9.7,color=".24")
    return f


def strategy(p,t):
    f=canvas(10,"Ретроспективный бэктест. Вход после публикации EIA, условный выход в четвёртую или пятую сессию.\nИсторическое правило отобрано после сравнения вариантов. Даты на кривой относятся к закрытию сделок.")
    a=f.add_axes([.104,.314,.837,.526])
    x0=pd.Timestamp(p.release_date.min())
    xend=p.exit_date.max()
    boundaries=[pd.Timestamp("2023-01-01"),pd.Timestamp("2024-01-01")]
    chart_axis(a,major=15)
    a.set_ylim(-9,87)
    a.set_xlim(x0-pd.Timedelta(days=31),xend+pd.Timedelta(days=90))
    a.set_ylabel("Накопленный валовой результат, тыс. долл. США",fontsize=10.6,labelpad=7)
    a.axhline(0,color=".42",lw=.6)
    a.axvspan(boundaries[0],boundaries[1],color=".95",zorder=0)
    for boundary in boundaries:a.axvline(boundary,c=".37",lw=.73,ls=(0,(4,4)),zorder=1)
    a.step([x0,*t.exit_date],[0,*(t.cumulative_gross_usd/1000)],where="post",color=".025",lw=1.42,zorder=4)
    a.scatter([xend],[t.cumulative_gross_usd.iloc[-1]/1000],s=18,c="black",zorder=5)
    for left,right,title,amount in [(x0,boundaries[0],"Разработка","+9 380 долл."),(boundaries[0],boundaries[1],"Валидация","+2 550 долл."),
                                    (boundaries[1],xend,"Историческая проверка","+62 050 долл.")]:
        at=left+(right-left)*.5
        a.text(at,83,title,ha="center",va="top",fontsize=10.6,weight="bold")
        a.text(at,76,amount,ha="center",va="top",fontsize=10.1)
    f.text(.103,.902,"24.01.2019–11.11.2026  |  389 окон, 70 закрытых сделок",fontsize=11.3)
    f.text(.940,.902,"Итог  +73 980 долл.",ha="right",fontsize=11.3,weight="bold")
    f.text(.105,.252,"Валовой результат по году выпуска EIA",fontsize=10.8,weight="bold",color=".17")
    table=f.add_axes([.102,.145,.84,.084]);table.set_axis_off()
    years=[str(y) for y in range(2019,2027)]
    annual=t.groupby("calendar_year").gross_usd.agg(["sum","size"]).reindex(years,fill_value=0)
    table.plot([0,1],[.56,.56],transform=table.transAxes,color=".45",lw=.55)
    for i,y in enumerate(years):
        at=(i+.5)/len(years); v=int(annual.loc[y,"sum"])
        label=f"{v:+,}".replace(","," ").replace("-","−") if v else "0"
        table.text(at,.80,y,transform=table.transAxes,ha="center",va="center",fontsize=9.9)
        table.text(at,.29,label,transform=table.transAxes,ha="center",va="center",fontsize=10.05)
    return f



def ml_quality_detail():
    f=canvas(11,"Отдельный инженерный эксперимент из train_shadow_quality.py. Метка проверяет согласие двух оценок.\nОбе оценки могут иметь общий сдвиг. Для абсолютной высоты используют отдельные контрольные данные.")
    a=drawing(f)
    group(a,1,10,36,44,"11 входных признаков")
    group(a,42,10,28,44,"Оценка вероятности")
    group(a,75,10,24,44,"Разметка и контроль")
    box(a,3,39,32,12,"Ошибка и форма минимума",
        "выигрыш над плоской моделью\nи обратным Солнцем, ширина минимума",body_size=9.7)
    box(a,3,26,32,10,"Контраст и допуски",
        "контраст к шуму, запас до границы $g$,\nостаток подгонки, число пикселей",body_size=9.6)
    box(a,3,13,32,10,"Геометрия и регистрация",
        "диаметр, высота Солнца, зенит обзора,\nвеличина сдвига при совмещении",body_size=9.6)
    box(a,44,40,24,11,"Стандартизация",
        r"$z_j=(x_j-\mu_j)/s_j$" + "\nпо обучающим примерам",body_size=9.8)
    box(a,44,27,24,9,"Линейный логит",r"$a=b+\sum_j w_j z_j$",shade=".94",body_size=11.3)
    box(a,44,13,24,10,"Оценка согласия",
        r"$p=1/(1+\exp(-a))$",shade=".91",body_size=11.2)
    for yy in (45,31,18):
        line(a,((35,yy),(39,yy)),lw=.65)
    arrow(a,(39,18),(39,45),(44,45),lw=.69)
    arrow(a,(56,40),(56,36));arrow(a,(56,27),(56,23))
    box(a,77,40,20,11,"Целевая метка",
        r"$y=1$ при $|\widehat g_1-\widehat g_2|\leq 2$ м" + "\nв близких пролётах",size=10.5,body_size=9.35)
    box(a,77,27,20,9,"Обучение",
        "логарифмическая потеря\nL2 = 0,05 для весов",body_size=9.35)
    box(a,77,13,20,10,"Проверка",
        "порог по исключённым\nдатам пролётов",body_size=9.35)
    arrow(a,(87,40),(87,36),dash=True)
    arrow(a,(77,31.5),(68,31.5),dash=True)
    arrow(a,(68,18),(77,18),dash=True)
    write_text(a,51,4.4,"На проверочной дате 90 из 94 пар с наибольшим баллом согласовались в пределах 2 м. По физическим диагностикам — 73 из 94.",10.15,italic=True)
    return f


def ru(value, decimals=0):
    return f"{value:.{decimals}f}".replace("-", "−").replace(".", ",")


def registration_chart():
    f=canvas(12,"Контрольные участки отложенных пар не использовались при подборе весов. Ошибку сначала усредняли внутри каждой пары,\nзатем по 116 парам. Уменьшение ошибки совмещения само по себе не оценивает точность прогноза запасов.")
    a=drawing(f)
    group(a,1,27,98,29,"(а) Как определяется смещение")
    box(a,3,41,23,11,"Два снимка B04","неподвижные участки\nвне резервуаров")
    box(a,30,41,23,11,"Четыре смещения","грубое, медианное,\nлокальное, линейное")
    box(a,57,41,18,11,"Модель весов",r"$\widehat d=w_1d_1+\cdots+w_4d_4$" + "\nсумма весов равна 1",shade=".93",body_size=9.6)
    box(a,79,41,18,11,"Контроль сдвига","выбор ближайшего\nдопустимого варианта",body_size=9.65)
    box(a,30,30,45,7,"Обучение на прошлых парах и контрольных участках",shade=".96",size=9.8)
    for left,right in ((26,30),(53,57),(75,79)): arrow(a,(left,46.5),(right,46.5))
    arrow(a,(64.6,37),(64.6,41),dash=True)
    write_text(a,1.7,23.5,"(б) Проверка на 116 парах снимков и 5 218 контрольных участках",11.7,bold=True,align="left")
    rows=read("shadow_registration_ml_patch_checks").query('study_phase=="holdout"')
    modes=("coarse","spatial","local","scene","ml")
    cols=[f"{key}_error_m" for key in modes]
    grouped=rows.groupby(["earlier_version","later_version"],sort=False)[cols].mean()
    summary=json.loads((DERIVED/"shadow_registration_ml_summary.json").read_text())["exterior_checks"]["holdout"]
    assert len(rows)==summary["patches"]==5218 and len(grouped)==summary["scenes"]==116
    expected=summary["scene_balanced_mean_error_m"]
    assert np.allclose(grouped.mean().to_numpy(),[expected[key] for key in modes])
    plot=f.add_axes([.296,.215,.613,.218])
    names=("Грубое смещение","Линейная модель","Локальное среднее","Медиана по сцене","Обученная комбинация")
    plot.spines[["right","top","left"]].set_visible(False)
    plot.set(xlim=(1.125,1.425),ylim=(-.55,4.55))
    yy=np.arange(4,-1,-1)
    plot.xaxis.set_major_locator(MultipleLocator(.05))
    plot.xaxis.set_major_formatter(FuncFormatter(lambda x,_:ru(x,2)))
    plot.set_yticks(yy,names,fontsize=10.5)
    plot.tick_params(axis="y",length=0,pad=10)
    plot.tick_params(axis="x",width=.6,length=3,labelsize=9.45)
    plot.set_xlabel("Средняя ошибка совмещения, м",fontsize=9.8,labelpad=4)
    plot.grid(axis="x",color=".85",lw=.5)
    for y,key in zip(yy,modes,strict=True):
        v=expected[key]
        plot.hlines(y,1.125,v,color=".48",lw=.77)
        plot.scatter(v,y,s=18 if key!="ml" else 27,c="black",zorder=4,marker="o" if key!="ml" else "D")
        plot.annotate(ru(v,3),(v,y),xytext=(8,0),textcoords="offset points",va="center",fontsize=9.6)
    return f, grouped.reset_index()


TERMINAL_LABELS={
    "plains_cushing":"Plains",
    "magellan_cushing":"Бывший Magellan",
    "enterprise_cushing":"Enterprise",
    "semgroup_cushing":"Бывший SemGroup",
    "ngl_cushing":"NGL",
    "enbridge_cushing":"Enbridge",
    "enbridge_blueknight":"Бывший Blueknight",
    "amid_cushing":"AMID",
    "phillips66_cushing":"Phillips 66",
    "cushing_connect":"Cushing Connect",
}


def consensus_chart():
    f=canvas(13,"Выпуск 13.03.2024 показывает расчёт по отдельным площадкам. Их локальные оценки не складываются напрямую.\nНа 148 выпусках дополнительная поправка по согласованности не снизила среднюю абсолютную ошибку.")
    a=drawing(f)
    group(a,1,34,98,22,"(а) Из отдельных крыш к признакам согласованности")
    box(a,3,38,28,14,"Крыши одного терминала","изменение высоты, площадь,\nдоли недели и общая ошибка сцены",body_size=9.8)
    box(a,36,38,28,14,"Оптическая оценка терминала","поправка и информативность $I_j$\nс учётом совместной погрешности",shade=".94",size=10.5,body_size=9.8)
    box(a,69,38,28,14,"Сводка по Кушингу","согласие знаков, разброс и доля\nсамого информативного терминала",body_size=9.7)
    arrow(a,(31,45),(36,45));arrow(a,(64,45),(69,45))
    write_text(a,1.5,29.6,"(б) Локальные результаты для выпуска 13.03.2024",11.7,bold=True,align="left")
    sample=read("cushing_terminal_consensus_terminals").query('release_date=="2024-03-13"').sort_values("information",ascending=False).copy()
    result=read("cushing_terminal_consensus_forecasts").query('release_date=="2024-03-13"').iloc[0]
    summary=json.loads((DERIVED/"cushing_terminal_consensus_summary.json").read_text())["accuracy"]["holdout"]["all"]
    assert summary["windows"]==148 and summary["models"]["learned_breadth"]["mae_kbbl"]>=summary["models"]["long_median_and_regional"]["mae_kbbl"]
    sample["information_share"]=sample.information/sample.information.sum()
    sample["terminal_label"]=sample.terminal_id.map(TERMINAL_LABELS)
    top_share=sample.information_share.max()
    effective=1/np.square(sample.information_share).sum()
    assert len(sample)==result.terminals and np.isclose(top_share,result.dominant_terminal_share)
    assert np.isclose(effective,result.effective_terminals) and sample.terminal_label.notna().all()

    yy=np.arange(len(sample)-1,-1,-1)
    bar=f.add_axes([.188,.208,.417,.315])
    wgt=f.add_axes([.677,.208,.251,.315],sharey=bar)
    for p in (bar,wgt):
        p.spines[["right","top","left"]].set_visible(False)
        p.tick_params(axis="y",length=0,pad=10)
        p.tick_params(axis="x",labelsize=9.2,length=2.9,width=.57)
        p.grid(axis="x",lw=.55,color=".86")
        p.set_ylim(-.57,9.58)
    bar.set_yticks(yy,sample.terminal_label,fontsize=9.55)
    bar.set_xlim(-.48,.104)
    bar.xaxis.set_major_locator(MultipleLocator(.1))
    bar.xaxis.set_major_formatter(FuncFormatter(lambda x,_:ru(x,1)))
    bar.set_xlabel("Локальная поправка, млн баррелей",fontsize=9.55,labelpad=3)
    bar.axvline(0,color=".28",lw=.65)
    bar.barh(yy,sample.correction_mbbl,height=.55,color=".26",zorder=3)
    wgt.set_xlim(0,19)
    wgt.xaxis.set_major_locator(MultipleLocator(5))
    wgt.tick_params(labelleft=False)
    wgt.set_xlabel("Доля информативности, %",fontsize=9.55,labelpad=3)
    wgt.barh(yy,100*sample.information_share,height=.55,facecolor=".87",edgecolor=".17",lw=.48,zorder=3)
    f.text(.933,.552,
           f"{int(sample.roofs.sum())} крыш, {len(sample)} площадок, эффективное число площадок {ru(effective,2)}",
           ha="right",fontsize=9.65)
    return f, sample


def optical_revision_chart(q):
    f=canvas(14,"Ретроспективное сравнение одних и тех же 148 выпусков. Базовый прогноз сохраняет остальные компоненты модели.\nПунктир слева отмечает нулевое изменение ошибки. Стрелки справа показывают сдвиг прогноза.")
    data=q[["release_date","actual","base","same_week","update"]].copy()
    data["optical_correction_kbbl"]=1000*data["update"]
    data["error_reduction_kbbl"]=1000*((data.actual-data.base).abs()-(data.actual-data.same_week).abs())
    active=data.optical_correction_kbbl.abs()>1e-7
    improved=data.error_reduction_kbbl>1e-7
    worse=data.error_reduction_kbbl<-1e-7
    assert active.sum()==101 and improved.sum()==50 and worse.sum()==51
    assert np.allclose(data.base+data["update"],data.same_week)
    scatter=f.add_axes([.105,.248,.467,.563])
    for spine in ("right","top"):scatter.spines[spine].set_visible(False)
    scatter.set_title("(а) Поправка и изменение ошибки",loc="left",pad=13,fontsize=11.9,weight="bold")
    scatter.set(xlim=(-755,755),ylim=(-620,750))
    scatter.xaxis.set_major_locator(FixedLocator([-600,-300,0,300,600]))
    scatter.yaxis.set_major_locator(FixedLocator([-600,-300,0,300,600]))
    for ax in (scatter.xaxis,scatter.yaxis):ax.set_major_formatter(FuncFormatter(lambda x,_:ru(x)))
    scatter.axhline(0,color=".33",lw=.72,ls=(0,(4,3)))
    scatter.axvline(0,color=".70",lw=.52)
    scatter.grid(color=".88",lw=.52)
    scatter.scatter(data.loc[improved,"optical_correction_kbbl"],data.loc[improved,"error_reduction_kbbl"],
                    s=16,color=".12",lw=0,label="Ошибка уменьшилась (50)",zorder=4)
    scatter.scatter(data.loc[worse,"optical_correction_kbbl"],data.loc[worse,"error_reduction_kbbl"],
                    s=18,fc="white",ec=".12",lw=.65,label="Ошибка выросла (51)",zorder=4)
    scatter.tick_params(axis="both",length=3,labelsize=9.35)
    scatter.set_xlabel("Спутниковая поправка, тыс. баррелей",fontsize=10.25)
    scatter.set_ylabel("Уменьшение абсолютной ошибки, тыс. баррелей",fontsize=10.25,labelpad=8)
    scatter.legend(frameon=False,fontsize=8.8,loc="upper right",borderpad=.45,handletextpad=.45)
    f.text(.105,.153,"Ещё на 47 выпусках текущая оптическая поправка равна нулю.",fontsize=9.65)
    f.text(.616,.834,"(б) Два выпуска в одной шкале",fontsize=11.9,weight="bold")
    examples=("2024-03-13","2025-05-07")
    for yy, issue in zip((.551,.314),examples,strict=True):
        row=data[data.release_date.eq(issue)].iloc[0]
        a=f.add_axes([.626,yy,.310,.171])
        a.set(xlim=(-1,1),ylim=(0,1))
        a.set_yticks([])
        a.xaxis.set_major_locator(FixedLocator([-1,-.5,0,.5,1]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x,_:ru(x,1).removesuffix(",0")))
        for spine in ("right","left","top"):a.spines[spine].set_visible(False)
        a.spines["bottom"].set_position(("data",.275))
        a.tick_params(axis="x",labelsize=9.0,length=2.5,width=.55)
        a.axvline(0,ymin=.275,ymax=.88,c=".82",lw=.56)
        a.vlines(row.actual,.275,.845,colors=".18",lw=.7,linestyles=(0,(3,2)))
        a.scatter(row.actual,.80,s=19,marker="D",c="black",zorder=5)
        arrow(a,(row.base,.54),(row.same_week,.54),lw=.83,z=4)
        a.scatter(row.base,.54,s=17,fc="white",ec="black",lw=.65,zorder=6)
        effect="снизилась" if row.error_reduction_kbbl>0 else "выросла"
        amount=ru(abs(row.error_reduction_kbbl))
        a.text(0,1.18,f"{pd.Timestamp(issue):%d.%m.%Y}  |  ошибка {effect} на {amount} тыс.",
               transform=a.transAxes,fontsize=10.1,va="bottom")
    f.text(.781,.305,"Недельное изменение, млн баррелей",fontsize=9.6,ha="center")
    key=f.add_axes([.617,.174,.328,.085])
    key.set(xlim=(0,1),ylim=(0,1))
    key.axis("off")
    key.scatter(.035,.74,s=17,fc="white",ec="black",lw=.68)
    write_text(key,.063,.74,"Без поправки",9.55,align="left")
    arrow(key,(.405,.74),(.471,.74),lw=.76)
    write_text(key,.50,.74,"С поправкой",9.55,align="left")
    key.scatter(.035,.22,s=21,c="black",marker="D")
    write_text(key,.063,.22,"Первая публикация EIA",9.55,align="left")
    return f, data


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    eia=largest_eia()
    forecast,stats=forecast_values(eia)
    prices,trades=market_ledger()
    tanks=map_inventory()
    registration, registration_data=registration_chart()
    consensus, consensus_data=consensus_chart()
    revision, revision_data=optical_revision_chart(forecast)
    figures=(architecture,shadow_physics,training,roof_tank,physical_model,lambda:cushing_map(tanks),
             lambda:eia_inventory(eia),lambda:forecast_eia(forecast,stats),lambda:wti_prices(prices),
             lambda:strategy(prices,trades), ml_quality_detail,
             lambda:registration, lambda:consensus, lambda:revision)
    meta={"Title":"Спутниковая оценка запасов Кушинга и проверка прогноза","Subject":"Иллюстрации к исследованию"}
    with PdfPages(OUT/"Cushing-research-figures.pdf",metadata=meta) as book:
        for (stem,title),make in zip(NAMES,figures,strict=True):
            fig=make()
            book.savefig(fig)
            fig.savefig(OUT/f"{stem}.png",dpi=300)
            print(f"{stem}  {title}",flush=True)
            plt.close(fig)
    tanks.to_csv(OUT/"06-map-tanks.csv",index=False)
    registration_data.to_csv(OUT/"12-registration-data.csv",index=False)
    consensus_data.to_csv(OUT/"13-terminal-data.csv",index=False)
    revision_data.to_csv(OUT/"14-optical-revision-data.csv",index=False)
    eia.drop(columns="date").to_csv(OUT/"07-eia-data.csv",index=False)
    forecast[["release_date","week_ending","cutoff","known","actual","same_week","no_revision",
              "reversed_direction","observed_terminals"]].to_csv(OUT/"08-forecast-data.csv",index=False)
    prices.drop(columns=["entry_date","exit_date"]).to_csv(OUT/"09-wti-data.csv",index=False)
    trades.drop(columns=["entry_date","exit_date"]).to_csv(OUT/"10-strategy-data.csv",index=False)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cushing-fonts-") as directory:
        register_cmu(Path(directory))
        main()
