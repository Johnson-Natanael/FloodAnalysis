import ee
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import random
random.seed(42)
np.random.seed(42)


# -------------------------
# 1) Inisialisasi dan Autentikasi Earth Engine
# Parameter project disesuaikan dengan nama akun GEE
# -------------------------
try:
    ee.Initialize(project="ee-johnsonnn")  
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

# -------------------------
# 2) Penentuan Area Studi
# -------------------------
area = ee.Geometry.Polygon(
    [[[106.58696441861906, -5.993917699906236],
    [106.58696441861906, -6.050253340255448],
    [106.63468628141203, -6.050253340255448],
    [106.63468628141203, -5.993917699906236]]]
)

# -------------------------
# 3) Pengumpulan Data Citra Sentinel-2 dan Preprocessing
# -------------------------
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

def add_ndwi(image):
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands(ndwi)

startDate = '2024-01-01'
endDate = '2025-01-01'

image = (s2
         .filterDate(startDate, endDate)
         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
         .filterBounds(area)
         .map(add_ndwi)
         .median()
         .clip(area)
        )

# aplikasi kernel gaussian untuk mengurangi noise
kernel = ee.Kernel.gaussian(radius=4, sigma=0.7, units='pixels')
image = image.convolve(kernel)

# -------------------------
# 4) Penentuan Titik Training Air dan Darat
# -------------------------
water = ee.FeatureCollection([
    ee.Feature(
            ee.Geometry.Point([106.58889560910978, -5.996258024372027]),
            {
              "Class": 1,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60271434995451, -5.996599468604145]),
            {
              "Class": 1,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61181240293303, -5.996599468604145]),
            {
              "Class": 1,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61824970456877, -5.997282356427141]),
            {
              "Class": 1,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60563259336271, -6.002062547240991]),
            {
              "Class": 1,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59593372556486, -5.997794521733255]),
            {
              "Class": 1,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59292965146818, -5.996258024372027]),
            {
              "Class": 1,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59026890012541, -6.003684388170614]),
            {
              "Class": 1,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61381295947098, -6.002574708056084]),
            {
              "Class": 1,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61844781664871, -6.0047087062701925]),
            {
              "Class": 1,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60660318163895, -6.017597877630029]),
            {
              "Class": 1,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59201196459793, -6.027157860319506]),
            {
              "Class": 1,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62057602438156, -6.013961756164251]),
            {
              "Class": 1,
              "system:index": "12"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6211768392009, -6.003889382339228]),
            {
              "Class": 1,
              "system:index": "13"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59126227169513, -6.013336512022581]),
            {
              "Class": 1,
              "system:index": "14"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59778540401935, -6.007019953139978]),
            {
              "Class": 1,
              "system:index": "15"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60353606014728, -6.009751447059137]),
            {
              "Class": 1,
              "system:index": "16"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61289160519122, -6.00923929299336]),
            {
              "Class": 1,
              "system:index": "17"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60576765804767, -6.00522740280822]),
            {
              "Class": 1,
              "system:index": "18"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60113280086993, -6.00095940208549]),
            {
              "Class": 1,
              "system:index": "19"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59443800716876, -6.002154445657695]),
            {
              "Class": 1,
              "system:index": "20"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59495299129962, -6.007105312532408]),
            {
              "Class": 1,
              "system:index": "21"
            }),
        ee.Feature(
            ee.Geometry.Point([106.595382144742, -6.01342187042423]),
            {
              "Class": 1,
              "system:index": "22"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59521048336505, -6.019396925240925]),
            {
              "Class": 1,
              "system:index": "23"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59194891720294, -6.018457992404979]),
            {
              "Class": 1,
              "system:index": "24"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58903067379474, -6.01393402055285]),
            {
              "Class": 1,
              "system:index": "25"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58911650448321, -6.006763874882423]),
            {
              "Class": 1,
              "system:index": "26"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58860152035236, -6.028444740292892]),
            {
              "Class": 1,
              "system:index": "27"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59091894894122, -6.021189428876656]),
            {
              "Class": 1,
              "system:index": "28"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58791487484454, -6.019311567777424]),
            {
              "Class": 1,
              "system:index": "29"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59040396481036, -6.023237997217345]),
            {
              "Class": 1,
              "system:index": "30"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59898703365802, -6.021872285848769]),
            {
              "Class": 1,
              "system:index": "31"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60508101253986, -6.022299071020388]),
            {
              "Class": 1,
              "system:index": "32"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61074583797931, -6.021530857470025]),
            {
              "Class": 1,
              "system:index": "33"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61143248348712, -6.018116561880607]),
            {
              "Class": 1,
              "system:index": "34"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61641066341876, -6.011543982493723]),
            {
              "Class": 1,
              "system:index": "35"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61563818722247, -6.016238690162712]),
            {
              "Class": 1,
              "system:index": "36"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61460821896075, -6.019396925240925]),
            {
              "Class": 1,
              "system:index": "37"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61168997555255, -6.02221371401289]),
            {
              "Class": 1,
              "system:index": "38"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6078275945711, -6.022555141962375]),
            {
              "Class": 1,
              "system:index": "39"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60817091732501, -6.011970775774409]),
            {
              "Class": 1,
              "system:index": "40"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60139029293536, -6.014104737155213]),
            {
              "Class": 1,
              "system:index": "41"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59950201778888, -6.009580729090764]),
            {
              "Class": 1,
              "system:index": "42"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60113280086993, -6.0050566834212935]),
            {
              "Class": 1,
              "system:index": "43"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59847204952716, -6.001386203662092]),
            {
              "Class": 1,
              "system:index": "44"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59898703365802, -5.997288894732929]),
            {
              "Class": 1,
              "system:index": "45"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62310545711993, -5.995752395947322]),
            {
              "Class": 1,
              "system:index": "46"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62233298092364, -5.999593634794571]),
            {
              "Class": 1,
              "system:index": "47"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62404959469318, -6.0036055664720696]),
            {
              "Class": 1,
              "system:index": "48"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6254228857088, -6.007446749968266]),
            {
              "Class": 1,
              "system:index": "49"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62087385921954, -6.007190671911442]),
            {
              "Class": 1,
              "system:index": "50"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61984389095782, -6.010007523911307]),
            {
              "Class": 1,
              "system:index": "51"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62404959469318, -6.009751447059137]),
            {
              "Class": 1,
              "system:index": "52"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62087385921954, -6.015129035643976]),
            {
              "Class": 1,
              "system:index": "53"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61761229305743, -6.0167508376382415]),
            {
              "Class": 1,
              "system:index": "54"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61495154171466, -6.020421213756969]),
            {
              "Class": 1,
              "system:index": "55"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60422270565509, -6.023494067716585]),
            {
              "Class": 1,
              "system:index": "56"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59855788021564, -6.024006208352777]),
            {
              "Class": 1,
              "system:index": "57"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59546797543048, -6.0256279838468245]),
            {
              "Class": 1,
              "system:index": "58"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59855788021564, -6.0261401224699025]),
            {
              "Class": 1,
              "system:index": "59"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59426634579181, -6.021872285848769]),
            {
              "Class": 1,
              "system:index": "60"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60139029293536, -6.017006911195064]),
            {
              "Class": 1,
              "system:index": "61"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60190527706622, -6.020165141808999]),
            {
              "Class": 1,
              "system:index": "62"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60139029293536, -6.024603705151099]),
            {
              "Class": 1,
              "system:index": "63"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58963148861407, -6.029981146736463]),
            {
              "Class": 1,
              "system:index": "64"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6104025152254, -6.005910279820817]),
            {
              "Class": 1,
              "system:index": "65"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60963003902911, -6.000361879316769]),
            {
              "Class": 1,
              "system:index": "66"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60679762630939, -5.997203533802827]),
            {
              "Class": 1,
              "system:index": "67"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60327856808185, -6.002325165954096]),
            {
              "Class": 1,
              "system:index": "68"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60302107601642, -6.0050566834212935]),
            {
              "Class": 1,
              "system:index": "69"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59194891720294, -6.008471060991098]),
            {
              "Class": 1,
              "system:index": "70"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59486716061114, -6.0096660880816435]),
            {
              "Class": 1,
              "system:index": "71"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59220640926837, -6.001300843373508]),
            {
              "Class": 1,
              "system:index": "72"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59006064205646, -5.998483946351287]),
            {
              "Class": 1,
              "system:index": "73"
            }),
        ee.Feature(
            ee.Geometry.Point([106.615552356534, -5.997630338319687]),
            {
              "Class": 1,
              "system:index": "74"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61744063168048, -6.001044762427545]),
            {
              "Class": 1,
              "system:index": "75"
            }),
        ee.Feature(
            ee.Geometry.Point([106.615552356534, -6.003520206531146]),
            {
              "Class": 1,
              "system:index": "76"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61683981686114, -6.006251718006014]),
            {
              "Class": 1,
              "system:index": "77"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61374991207599, -6.005654201041435]),
            {
              "Class": 1,
              "system:index": "78"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61023085384845, -6.002154445657695]),
            {
              "Class": 1,
              "system:index": "79"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61126082211017, -6.010861112548243]),
            {
              "Class": 1,
              "system:index": "80"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61246245174884, -6.013848662231567]),
            {
              "Class": 1,
              "system:index": "81"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61907141476154, -6.011970775774409]),
            {
              "Class": 1,
              "system:index": "82"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62319128780841, -6.010690394927974]),
            {
              "Class": 1,
              "system:index": "83"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62035887508868, -5.997801060032897]),
            {
              "Class": 1,
              "system:index": "84"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62027304440021, -6.000105797929675]),
            {
              "Class": 1,
              "system:index": "85"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62473624020099, -5.996691367941721]),
            {
              "Class": 1,
              "system:index": "86"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62585203915118, -5.998825389189776]),
            {
              "Class": 1,
              "system:index": "87"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62722533016681, -6.000788681361268]),
            {
              "Class": 1,
              "system:index": "88"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63018648891925, -6.005696880846362]),
            {
              "Class": 1,
              "system:index": "89"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6303581502962, -6.007660148256956]),
            {
              "Class": 1,
              "system:index": "90"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63113062649249, -6.009410011068829]),
            {
              "Class": 1,
              "system:index": "91"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63246100216388, -6.011543982493723]),
            {
              "Class": 1,
              "system:index": "92"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63306181698321, -6.013720624724535]),
            {
              "Class": 1,
              "system:index": "93"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63263266354083, -6.01504367751027]),
            {
              "Class": 1,
              "system:index": "94"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63160269527911, -6.017006911195064]),
            {
              "Class": 1,
              "system:index": "95"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63306181698321, -6.020848000068688]),
            {
              "Class": 1,
              "system:index": "96"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6309160497713, -6.037254922042485]),
            {
              "Class": 1,
              "system:index": "97"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63310473232745, -6.027524404908733]),
            {
              "Class": 1,
              "system:index": "98"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63151686459064, -6.019160709104952]),
            {
              "Class": 1,
              "system:index": "99"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62783877968415, -6.001810328826878]),
            {
              "Class": 1,
              "system:index": "100"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62972705483064, -6.0040296899093555]),
            {
              "Class": 1,
              "system:index": "101"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62483470558747, -5.997755703512118]),
            {
              "Class": 1,
              "system:index": "102"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62612216591462, -5.999462917846499]),
            {
              "Class": 1,
              "system:index": "103"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6337354135394, -6.025059380394746]),
            {
              "Class": 1,
              "system:index": "104"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63107466219662, -6.03544794420163]),
            {
              "Class": 1,
              "system:index": "105"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63326334475278, -6.029899845231098]),
            {
              "Class": 1,
              "system:index": "106"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63279127596616, -6.031265536398839]),
            {
              "Class": 1,
              "system:index": "107"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58872394773283, -5.9995017359461]),
            {
              "Class": 1,
              "system:index": "108"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58872394773283, -6.002147907410265]),
            {
              "Class": 1,
              "system:index": "109"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59267215940275, -5.997623800017997]),
            {
              "Class": 1,
              "system:index": "110"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5950754186801, -5.998392047315774]),
            {
              "Class": 1,
              "system:index": "111"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59928112241545, -5.998135965003446]),
            {
              "Class": 1,
              "system:index": "112"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60374431821623, -5.997623800017997]),
            {
              "Class": 1,
              "system:index": "113"
            }),
        ee.Feature(
            ee.Geometry.Point([106.606405069559, -5.99881885090241]),
            {
              "Class": 1,
              "system:index": "114"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60829334470549, -6.0012089448128645]),
            {
              "Class": 1,
              "system:index": "115"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6086366674594, -6.004367267117687]),
            {
              "Class": 1,
              "system:index": "116"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60983829709807, -6.011708161713541]),
            {
              "Class": 1,
              "system:index": "117"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59267215940275, -6.004281907296112]),
            {
              "Class": 1,
              "system:index": "118"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59267215940275, -6.00573302244354]),
            {
              "Class": 1,
              "system:index": "119"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59009723874846, -6.008208445241393]),
            {
              "Class": 1,
              "system:index": "120"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60254268857756, -5.998050604205935]),
            {
              "Class": 1,
              "system:index": "121"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60520343992033, -5.999245654154948]),
            {
              "Class": 1,
              "system:index": "122"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61327152463713, -5.997794521733255]),
            {
              "Class": 1,
              "system:index": "123"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60992412778654, -5.995831218781335]),
            {
              "Class": 1,
              "system:index": "124"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5968778631381, -6.00308686838978]),
            {
              "Class": 1,
              "system:index": "125"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5910413763217, -6.009659549924401]),
            {
              "Class": 1,
              "system:index": "126"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58872394773283, -6.003342948376107]),
            {
              "Class": 1,
              "system:index": "127"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5884664556674, -6.008123086021921]),
            {
              "Class": 1,
              "system:index": "128"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5991094610385, -6.002830788283102]),
            {
              "Class": 1,
              "system:index": "129"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6294935247592, -5.99506296787658]),
            {
              "Class": 1,
              "system:index": "130"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63258342954435, -5.99523369039335]),
            {
              "Class": 1,
              "system:index": "131"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59747867795744, -6.015037139417656]),
            {
              "Class": 1,
              "system:index": "132"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5991094610385, -6.010769215600892]),
            {
              "Class": 1,
              "system:index": "133"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6049459478549, -6.014524990327895]),
            {
              "Class": 1,
              "system:index": "134"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60734920713224, -6.007354852444444]),
            {
              "Class": 1,
              "system:index": "135"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59876613828459, -6.017427162124498]),
            {
              "Class": 1,
              "system:index": "136"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59310131284514, -6.014524990327895]),
            {
              "Class": 1,
              "system:index": "137"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61627559873381, -6.007952367542839]),
            {
              "Class": 1,
              "system:index": "138"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61730556699553, -6.018366096741414]),
            {
              "Class": 1,
              "system:index": "139"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62348537656584, -6.0047087062701925]),
            {
              "Class": 1,
              "system:index": "140"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6228845617465, -6.006928055544112]),
            {
              "Class": 1,
              "system:index": "141"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63026600095549, -5.996428746514803]),
            {
              "Class": 1,
              "system:index": "142"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63284092160978, -5.996940912622516]),
            {
              "Class": 1,
              "system:index": "143"
            })
])
land = ee.FeatureCollection([
    ee.Feature(
            ee.Geometry.Point([106.61886359848232, -6.037498845611285]),
            {
              "Class": 0,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62135268844814, -6.039077903574118]),
            {
              "Class": 0,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62345554031582, -6.043004189901354]),
            {
              "Class": 0,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62079478897304, -6.0479546842360055]),
            {
              "Class": 0,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6235413710043, -6.04654635856188]),
            {
              "Class": 0,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62126685775966, -6.043601665759006]),
            {
              "Class": 0,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62002231277675, -6.040870342168816]),
            {
              "Class": 0,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61864902176113, -6.0391632579274415]),
            {
              "Class": 0,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61560203232021, -6.03690136301961]),
            {
              "Class": 0,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6160311857626, -6.038565777171647]),
            {
              "Class": 0,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61564494766445, -6.0398887693875976]),
            {
              "Class": 0,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61448623337002, -6.0407849880845665]),
            {
              "Class": 0,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61225463546963, -6.041638528321513]),
            {
              "Class": 0,
              "system:index": "12"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61028052963466, -6.042022620989102]),
            {
              "Class": 0,
              "system:index": "13"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60847808517666, -6.043132220497817]),
            {
              "Class": 0,
              "system:index": "14"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60667564071865, -6.044583231807359]),
            {
              "Class": 0,
              "system:index": "15"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60959388412685, -6.044497878308612]),
            {
              "Class": 0,
              "system:index": "16"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61242629684658, -6.0436443425807385]),
            {
              "Class": 0,
              "system:index": "17"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61216880478115, -6.046076915855629]),
            {
              "Class": 0,
              "system:index": "18"
            }),
        ee.Feature(
            ee.Geometry.Point([106.611782566683, -6.048893565981237]),
            {
              "Class": 0,
              "system:index": "19"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60869266189785, -6.046759741475501]),
            {
              "Class": 0,
              "system:index": "20"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60616065658779, -6.04654635856188]),
            {
              "Class": 0,
              "system:index": "21"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60422946609707, -6.045479442731193]),
            {
              "Class": 0,
              "system:index": "22"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60204078354091, -6.044113787397681]),
            {
              "Class": 0,
              "system:index": "23"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59598972000332, -6.042065297935347]),
            {
              "Class": 0,
              "system:index": "24"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59581805862636, -6.043516312105441]),
            {
              "Class": 0,
              "system:index": "25"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5969767729208, -6.0417238822712]),
            {
              "Class": 0,
              "system:index": "26"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59727718033047, -6.04040089453894]),
            {
              "Class": 0,
              "system:index": "27"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58774997390957, -6.0383523910277015]),
            {
              "Class": 0,
              "system:index": "28"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59011031784267, -6.038309713788821]),
            {
              "Class": 0,
              "system:index": "29"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59264232315273, -6.0394193209067195]),
            {
              "Class": 0,
              "system:index": "30"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59483100570888, -6.0404862486837345]),
            {
              "Class": 0,
              "system:index": "31"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59435893692226, -6.041254435381426]),
            {
              "Class": 0,
              "system:index": "32"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59015323318691, -6.039974123613115]),
            {
              "Class": 0,
              "system:index": "33"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58796455063076, -6.03929128943222]),
            {
              "Class": 0,
              "system:index": "34"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60096789993496, -6.046418328773303]),
            {
              "Class": 0,
              "system:index": "35"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60328532852382, -6.047997360714339]),
            {
              "Class": 0,
              "system:index": "36"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60495902694912, -6.049106947968706]),
            {
              "Class": 0,
              "system:index": "37"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59925128616543, -6.04539408937384]),
            {
              "Class": 0,
              "system:index": "38"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61478664077968, -6.04863750748513]),
            {
              "Class": 0,
              "system:index": "39"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61495830215664, -6.045351412690105]),
            {
              "Class": 0,
              "system:index": "40"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62984992660732, -6.046375652170381]),
            {
              "Class": 0,
              "system:index": "41"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63062240280361, -6.0484241253123985]),
            {
              "Class": 0,
              "system:index": "42"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63289691604824, -6.023713901022511]),
            {
              "Class": 0,
              "system:index": "43"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63160945572109, -6.02751226437248]),
            {
              "Class": 0,
              "system:index": "44"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62864829696865, -6.033273775998929]),
            {
              "Class": 0,
              "system:index": "45"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6253438154623, -6.033700552202758]),
            {
              "Class": 0,
              "system:index": "46"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62508632339687, -6.031267923340842]),
            {
              "Class": 0,
              "system:index": "47"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60985137619228, -6.02836582551203]),
            {
              "Class": 0,
              "system:index": "48"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60916473068447, -6.030883823049208]),
            {
              "Class": 0,
              "system:index": "49"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60989429153652, -6.031908089889008]),
            {
              "Class": 0,
              "system:index": "50"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61294128097744, -6.026231920145692]),
            {
              "Class": 0,
              "system:index": "51"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61886359848232, -6.024354076483063]),
            {
              "Class": 0,
              "system:index": "52"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62023688949795, -6.024695503086714]),
            {
              "Class": 0,
              "system:index": "53"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62139560379238, -6.025292999126399]),
            {
              "Class": 0,
              "system:index": "54"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61903525985927, -6.02670138004625]),
            {
              "Class": 0,
              "system:index": "55"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6198506513998, -6.028493859567155]),
            {
              "Class": 0,
              "system:index": "56"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61392833389492, -6.028493859567155]),
            {
              "Class": 0,
              "system:index": "57"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61371375717373, -6.030030265871609]),
            {
              "Class": 0,
              "system:index": "58"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61414291061611, -6.03233486716789]),
            {
              "Class": 0,
              "system:index": "59"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6114392439291, -6.037285459047048]),
            {
              "Class": 0,
              "system:index": "60"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60890723861904, -6.0389071948271]),
            {
              "Class": 0,
              "system:index": "61"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60268451370449, -6.041041050296948]),
            {
              "Class": 0,
              "system:index": "62"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60547401107998, -6.041169081357721]),
            {
              "Class": 0,
              "system:index": "63"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60753394760341, -6.0415104973717675]),
            {
              "Class": 0,
              "system:index": "64"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60800601639004, -6.04065695693297]),
            {
              "Class": 0,
              "system:index": "65"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60989429153652, -6.03924861226732]),
            {
              "Class": 0,
              "system:index": "66"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61281253494472, -6.038309713788821]),
            {
              "Class": 0,
              "system:index": "67"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61422874130459, -6.037114749735139]),
            {
              "Class": 0,
              "system:index": "68"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61688949264736, -6.0466743883201515]),
            {
              "Class": 0,
              "system:index": "69"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61753322281093, -6.04863750748513]),
            {
              "Class": 0,
              "system:index": "70"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6189494291708, -6.049234977120764]),
            {
              "Class": 0,
              "system:index": "71"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61701823868007, -6.043132220497817]),
            {
              "Class": 0,
              "system:index": "72"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61697532333584, -6.044412524796417]),
            {
              "Class": 0,
              "system:index": "73"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62319804825039, -6.041041050296948]),
            {
              "Class": 0,
              "system:index": "74"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62650252975673, -6.0439430802386385]),
            {
              "Class": 0,
              "system:index": "75"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62723209060879, -6.047442566233367]),
            {
              "Class": 0,
              "system:index": "76"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62817622818203, -6.049021595183827]),
            {
              "Class": 0,
              "system:index": "77"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62474300064297, -6.048893565981237]),
            {
              "Class": 0,
              "system:index": "78"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62547256149502, -6.046845094617378]),
            {
              "Class": 0,
              "system:index": "79"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62547256149502, -6.0437296962141]),
            {
              "Class": 0,
              "system:index": "80"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62525798477382, -6.040998373269956]),
            {
              "Class": 0,
              "system:index": "81"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6270175138876, -6.041425143388443]),
            {
              "Class": 0,
              "system:index": "82"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62950660385341, -6.04065695693297]),
            {
              "Class": 0,
              "system:index": "83"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63040782608242, -6.0394193209067195]),
            {
              "Class": 0,
              "system:index": "84"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60431529678554, -6.041339789391658]),
            {
              "Class": 0,
              "system:index": "85"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60804893173427, -6.026402632883797]),
            {
              "Class": 0,
              "system:index": "86"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61006595291347, -6.025762459839067]),
            {
              "Class": 0,
              "system:index": "87"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58852245010586, -6.0346821361963725]),
            {
              "Class": 0,
              "system:index": "88"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5877928892538, -6.035834428181306]),
            {
              "Class": 0,
              "system:index": "89"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58899451889248, -6.036133170147801]),
            {
              "Class": 0,
              "system:index": "90"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58895160354824, -6.036816008309847]),
            {
              "Class": 0,
              "system:index": "91"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58817912735195, -6.041382466391728]),
            {
              "Class": 0,
              "system:index": "92"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58959533371181, -6.042364036464821]),
            {
              "Class": 0,
              "system:index": "93"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5914836088583, -6.042449390300108]),
            {
              "Class": 0,
              "system:index": "94"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63293983139248, -6.0383523910277015]),
            {
              "Class": 0,
              "system:index": "95"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63324023880214, -6.040443571613017]),
            {
              "Class": 0,
              "system:index": "96"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63349773086757, -6.042492067212703]),
            {
              "Class": 0,
              "system:index": "97"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63349773086757, -6.045138029220968]),
            {
              "Class": 0,
              "system:index": "98"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63366939224453, -6.0464610053728665]),
            {
              "Class": 0,
              "system:index": "99"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62963534988613, -6.042534744121925]),
            {
              "Class": 0,
              "system:index": "100"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62830497421474, -6.044241817731621]),
            {
              "Class": 0,
              "system:index": "101"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6277041593954, -6.038224359300984]),
            {
              "Class": 0,
              "system:index": "102"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62877704300136, -6.037370813682832]),
            {
              "Class": 0,
              "system:index": "103"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62817622818203, -6.036218524965118]),
            {
              "Class": 0,
              "system:index": "104"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62508632339687, -6.038139004799686]),
            {
              "Class": 0,
              "system:index": "105"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62379886306972, -6.036133170147801]),
            {
              "Class": 0,
              "system:index": "106"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62869121231289, -6.023500509034588]),
            {
              "Class": 0,
              "system:index": "107"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63023616470547, -6.023585865839828]),
            {
              "Class": 0,
              "system:index": "108"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63221027054043, -6.0238846145524585]),
            {
              "Class": 0,
              "system:index": "109"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63105155624599, -6.024396754820264]),
            {
              "Class": 0,
              "system:index": "110"
            })
])
training_fc = water.merge(land)

# -------------------------
# 5) Pemilihan Fitur dan Pembuatan Training Set serta Test Set
# -------------------------
bands = ['B5', 'B6', 'B7', 'B8', 'B11', 'NDWI']
input_image = image.select(bands)

sampled = input_image.sampleRegions(
    collection=training_fc,
    properties=['Class'],
    scale=10,
    geometries=True
)

# split menjadi train dan test set (80% train, 20% test)
sampled = sampled.randomColumn('random', seed=42)
trainSet = sampled.filter(ee.Filter.lt('random', 0.8))
testSet = sampled.filter(ee.Filter.greaterThanOrEquals('random', 0.8))

# -------------------------
# 6) Function untuk model prediksi Random Forest 
# Return: classifier -> hasil prediksi model
# -------------------------
def init_model():
    classifier = ee.Classifier.smileRandomForest(numberOfTrees=50, seed=42).train(
        features=trainSet,
        classProperty='Class',
        inputProperties=bands
    )
    return classifier

# -------------------------
# 7) Function untuk menggunakan model
# Return: filename -> path file hasil prediksi 
# -------------------------
def init_predict_sentinel (startDate, endDate):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    image = (s2
         .filterDate(startDate, endDate)
         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
         .filterBounds(area)
         .map(add_ndwi)
         .median()
         .clip(area))
    kernel = ee.Kernel.gaussian(radius=4, sigma=0.7, units='pixels')
    image = image.convolve(kernel)
    input_image = image.select(bands)
    
    classifier = init_model()
    classified = input_image.classify(classifier)
    
    classification_vis = {
        'min': 0,
        'max': 1,
        'palette': ['#b30326', '#3a4cc0']
    }

    # visualisasi dan generate thumbnail untuk download
    viz_image = classified.visualize(min=classification_vis['min'],
                                    max=classification_vis['max'],
                                    palette=classification_vis['palette'])

    region_geojson = area.getInfo()

    thumb_params = {
        'region': region_geojson,
        'dimensions': 1024,   
        'format': 'png'
    }

    thumb_url = viz_image.getThumbURL(thumb_params)
    resp = requests.get(thumb_url)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    
    # optionally save to disk
    # out_path = f"../web_app/static/assets/custom_model/prediction_{startDate}_{endDate}.png"
    # img.save(out_path)
    # print(f"Saved prediction image to {out_path}")
    
    # plot show prediction (raw)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(f"prediction {startDate} {endDate}")
    # plt.show()

    url = viz_image.getDownloadURL({
        'region': area,
        'scale': 30,
        'crs': 'EPSG:4326',
        'format': 'GEO_TIFF'
    })

    # download prediksi dalam format GeoTIFF
    response = requests.get(url)
    filename = f"../web_app/static/assets/custom_model/raw_data.tif"
    with open(filename, "wb") as f:
        f.write(response.content)
    
    return filename