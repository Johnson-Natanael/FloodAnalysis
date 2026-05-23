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
# 3) Pengumpulan Data Citra Landsat-8 dan Preprocessing
# -------------------------
l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")

def add_ndwi(image):
    ndwi = image.normalizedDifference(['B3', 'B5']).rename('NDWI')
    return image.addBands(ndwi)

startDate = '2024-01-01'
endDate = '2025-01-01'

image = (l8
         .filterDate(startDate, endDate)
         .filter(ee.Filter.lt('CLOUD_COVER', 20))
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
water = ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([106.63383461616283, -5.995516992068516]),
            {
              "Class": 1,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6335771240974, -5.998803388394944]),
            {
              "Class": 1,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63353420875316, -6.048609067689114]),
            {
              "Class": 1,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6319034256721, -6.047200743719052]),
            {
              "Class": 1,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63078762672191, -6.045322972721297]),
            {
              "Class": 1,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6319034256721, -6.042250242485792]),
            {
              "Class": 1,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63168884895092, -6.039369542048122]),
            {
              "Class": 1,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63104511878734, -6.037406389266157]),
            {
              "Class": 1,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63117386482006, -6.0349310995316685]),
            {
              "Class": 1,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63233257911449, -6.032754542320353]),
            {
              "Class": 1,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63336254737621, -6.030535298519812]),
            {
              "Class": 1,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63331963203197, -6.028316045638341]),
            {
              "Class": 1,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6330192246223, -6.026054105475693]),
            {
              "Class": 1,
              "system:index": "12"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6339204468513, -6.02182894671744]),
            {
              "Class": 1,
              "system:index": "13"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63409210822826, -6.024176261197945]),
            {
              "Class": 1,
              "system:index": "14"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63293339393383, -6.020719303613295]),
            {
              "Class": 1,
              "system:index": "15"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63147427222972, -6.019524300812493]),
            {
              "Class": 1,
              "system:index": "16"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63168884895092, -6.016920892754503]),
            {
              "Class": 1,
              "system:index": "17"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63271881721263, -6.014786942424472]),
            {
              "Class": 1,
              "system:index": "18"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63310505531078, -6.012396908113096]),
            {
              "Class": 1,
              "system:index": "19"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63239695213085, -6.0116286805685215]),
            {
              "Class": 1,
              "system:index": "20"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6316673912788, -6.010348298917063]),
            {
              "Class": 1,
              "system:index": "21"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63074471137767, -6.009003894940835]),
            {
              "Class": 1,
              "system:index": "22"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63035847327953, -6.007211351140627]),
            {
              "Class": 1,
              "system:index": "23"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63078762672191, -6.005418801438898]),
            {
              "Class": 1,
              "system:index": "24"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63119532249218, -6.003754285718812]),
            {
              "Class": 1,
              "system:index": "25"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63192488334423, -6.002303165300531]),
            {
              "Class": 1,
              "system:index": "26"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63271881721263, -6.000574619759259]),
            {
              "Class": 1,
              "system:index": "27"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63353420875316, -6.001406883111626]),
            {
              "Class": 1,
              "system:index": "28"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63366295478588, -6.002793985874533]),
            {
              "Class": 1,
              "system:index": "29"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63362003944164, -6.004117065219478]),
            {
              "Class": 1,
              "system:index": "30"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63319088599926, -6.005013342950047]),
            {
              "Class": 1,
              "system:index": "31"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6324613251472, -6.0057175601322745]),
            {
              "Class": 1,
              "system:index": "32"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6315601029182, -6.007104651918784]),
            {
              "Class": 1,
              "system:index": "33"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63226820609813, -6.003668925801195]),
            {
              "Class": 1,
              "system:index": "34"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58877350471263, -5.995965138188124]),
            {
              "Class": 1,
              "system:index": "35"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5919921555305, -5.996050499312035]),
            {
              "Class": 1,
              "system:index": "36"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59538246772533, -5.996007818751753]),
            {
              "Class": 1,
              "system:index": "37"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59903027198558, -5.995965138188124]),
            {
              "Class": 1,
              "system:index": "38"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6020772614265, -5.995965138188124]),
            {
              "Class": 1,
              "system:index": "39"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6055534043098, -5.995965138188124]),
            {
              "Class": 1,
              "system:index": "40"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60932995460277, -5.995965138188124]),
            {
              "Class": 1,
              "system:index": "41"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6174409546638, -5.996135860422563]),
            {
              "Class": 1,
              "system:index": "42"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62173248908763, -5.995922457621162]),
            {
              "Class": 1,
              "system:index": "43"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62580944679027, -5.995879777050868]),
            {
              "Class": 1,
              "system:index": "44"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62731148383861, -5.996861429322837]),
            {
              "Class": 1,
              "system:index": "45"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6255090393806, -5.998227203457724]),
            {
              "Class": 1,
              "system:index": "46"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62413574836498, -5.999635654451532]),
            {
              "Class": 1,
              "system:index": "47"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62323452613597, -6.002153785038078]),
            {
              "Class": 1,
              "system:index": "48"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62263371131664, -6.00420242506688]),
            {
              "Class": 1,
              "system:index": "49"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61975838325267, -6.005824259625902]),
            {
              "Class": 1,
              "system:index": "50"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61726929328685, -6.008043604353319]),
            {
              "Class": 1,
              "system:index": "51"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61568142555004, -6.009793465932525]),
            {
              "Class": 1,
              "system:index": "52"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61344982764965, -6.011799397894397]),
            {
              "Class": 1,
              "system:index": "53"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6107032456184, -6.013463889014128]),
            {
              "Class": 1,
              "system:index": "54"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60838581702953, -6.015043016906289]),
            {
              "Class": 1,
              "system:index": "55"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60512425086742, -6.015683202583228]),
            {
              "Class": 1,
              "system:index": "56"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6035363831306, -6.0174757184682415]),
            {
              "Class": 1,
              "system:index": "57"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58946015022045, -6.025883392627975]),
            {
              "Class": 1,
              "system:index": "58"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59113384864574, -6.0243469745825164]),
            {
              "Class": 1,
              "system:index": "59"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59362293861156, -6.02387751264578]),
            {
              "Class": 1,
              "system:index": "60"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59765698096996, -6.022981266003019]),
            {
              "Class": 1,
              "system:index": "61"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6015193619514, -6.02157287543306]),
            {
              "Class": 1,
              "system:index": "62"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59533955238109, -6.022853230647583]),
            {
              "Class": 1,
              "system:index": "63"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5995023407722, -6.021615553988843]),
            {
              "Class": 1,
              "system:index": "64"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6044376053596, -6.020121802541452]),
            {
              "Class": 1,
              "system:index": "65"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60696961066967, -6.018926798426484]),
            {
              "Class": 1,
              "system:index": "66"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60915829322582, -6.018371974194055]),
            {
              "Class": 1,
              "system:index": "67"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61143280647045, -6.017433039587293]),
            {
              "Class": 1,
              "system:index": "68"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61460854194408, -6.01602463463606]),
            {
              "Class": 1,
              "system:index": "69"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61632515571361, -6.014189434831283]),
            {
              "Class": 1,
              "system:index": "70"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61932922981029, -6.0116713599050025]),
            {
              "Class": 1,
              "system:index": "71"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62138916633373, -6.009366670944112]),
            {
              "Class": 1,
              "system:index": "72"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62319161079174, -6.007190011297932]),
            {
              "Class": 1,
              "system:index": "73"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62559487006908, -6.004927983229714]),
            {
              "Class": 1,
              "system:index": "74"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62589527747875, -6.002367185400464]),
            {
              "Class": 1,
              "system:index": "75"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62722565315013, -5.99942225301969]),
            {
              "Class": 1,
              "system:index": "76"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63005806586986, -5.99767235812794]),
            {
              "Class": 1,
              "system:index": "77"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63233257911449, -5.996776068325842]),
            {
              "Class": 1,
              "system:index": "78"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58941723487621, -6.021530196873931]),
            {
              "Class": 1,
              "system:index": "79"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59237839362865, -6.020420553159513]),
            {
              "Class": 1,
              "system:index": "80"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59632660529857, -6.019396264642236]),
            {
              "Class": 1,
              "system:index": "81"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59963108680492, -6.018115901280354]),
            {
              "Class": 1,
              "system:index": "82"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60186268470531, -6.017603755090991]),
            {
              "Class": 1,
              "system:index": "83"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5892884888435, -5.998782048222949]),
            {
              "Class": 1,
              "system:index": "84"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59267880103832, -5.9989100892424325]),
            {
              "Class": 1,
              "system:index": "85"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59521080634838, -5.9985259660937675]),
            {
              "Class": 1,
              "system:index": "86"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59958817146068, -5.998654007173389]),
            {
              "Class": 1,
              "system:index": "87"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60340763709789, -5.998867408905946]),
            {
              "Class": 1,
              "system:index": "88"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6064546265388, -5.998824728566113]),
            {
              "Class": 1,
              "system:index": "89"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58941723487621, -6.00185502439034]),
            {
              "Class": 1,
              "system:index": "90"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59280754707103, -6.0020257447805285]),
            {
              "Class": 1,
              "system:index": "91"
            }),
        ee.Feature(
            ee.Geometry.Point([106.595682875135, -6.001897704492903]),
            {
              "Class": 1,
              "system:index": "92"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59933067939525, -6.001726984062589]),
            {
              "Class": 1,
              "system:index": "93"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60400845191722, -6.001598943704775]),
            {
              "Class": 1,
              "system:index": "94"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60731293342357, -6.001470903316861]),
            {
              "Class": 1,
              "system:index": "95"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6112611450935, -5.998355244607497]),
            {
              "Class": 1,
              "system:index": "96"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61503769538646, -5.998654007173389]),
            {
              "Class": 1,
              "system:index": "97"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62031628272777, -5.998483285727218]),
            {
              "Class": 1,
              "system:index": "98"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61108948371654, -6.001641623827388]),
            {
              "Class": 1,
              "system:index": "99"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61465145728832, -6.001428223180877]),
            {
              "Class": 1,
              "system:index": "100"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61950089118724, -6.001556263578804]),
            {
              "Class": 1,
              "system:index": "101"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61748387000804, -5.999208851504289]),
            {
              "Class": 1,
              "system:index": "102"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58941723487621, -6.004501184427447]),
            {
              "Class": 1,
              "system:index": "103"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58933140418773, -6.006208377629879]),
            {
              "Class": 1,
              "system:index": "104"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58954598090892, -6.00893987562104]),
            {
              "Class": 1,
              "system:index": "105"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58946015022045, -6.0120981530857796]),
            {
              "Class": 1,
              "system:index": "106"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58950306556468, -6.014744263332448]),
            {
              "Class": 1,
              "system:index": "107"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58941723487621, -6.017689112822739]),
            {
              "Class": 1,
              "system:index": "108"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59375168464427, -6.007787526577224]),
            {
              "Class": 1,
              "system:index": "109"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59375168464427, -6.010732413728844]),
            {
              "Class": 1,
              "system:index": "110"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59379459998851, -6.013549247395803]),
            {
              "Class": 1,
              "system:index": "111"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5942237534309, -6.016792855970907]),
            {
              "Class": 1,
              "system:index": "112"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59735657356029, -6.004287784900899]),
            {
              "Class": 1,
              "system:index": "113"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59834362647777, -6.009878824890033]),
            {
              "Class": 1,
              "system:index": "114"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59808613441234, -6.0129944176755314]),
            {
              "Class": 1,
              "system:index": "115"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5980432190681, -6.016238029560181]),
            {
              "Class": 1,
              "system:index": "116"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60306431434398, -6.007446089355073]),
            {
              "Class": 1,
              "system:index": "117"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60289265296703, -6.013378530619068]),
            {
              "Class": 1,
              "system:index": "118"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60289265296703, -6.010262940033191]),
            {
              "Class": 1,
              "system:index": "119"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60744167945629, -6.00334882599102]),
            {
              "Class": 1,
              "system:index": "120"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60769917152172, -6.006421776403896]),
            {
              "Class": 1,
              "system:index": "121"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60739876411205, -6.009452029968564]),
            {
              "Class": 1,
              "system:index": "122"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60804249427562, -6.011485131433041]),
            {
              "Class": 1,
              "system:index": "123"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61134697578197, -6.003504034579903]),
            {
              "Class": 1,
              "system:index": "124"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61164738319164, -6.008753646214338]),
            {
              "Class": 1,
              "system:index": "125"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61156155250316, -6.010631543774977]),
            {
              "Class": 1,
              "system:index": "126"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61482311866527, -6.0029918746383295]),
            {
              "Class": 1,
              "system:index": "127"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61426521919017, -6.008156131995418]),
            {
              "Class": 1,
              "system:index": "128"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61739803931957, -6.00303455465185]),
            {
              "Class": 1,
              "system:index": "129"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62035919807201, -6.003290634662756]),
            {
              "Class": 1,
              "system:index": "130"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60885788581615, -5.99889457777796]),
            {
              "Class": 1,
              "system:index": "131"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6017339386726, -5.999620143007707]),
            {
              "Class": 1,
              "system:index": "132"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60087563178783, -6.004571032911747]),
            {
              "Class": 1,
              "system:index": "133"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6006181397224, -6.008028093148889]),
            {
              "Class": 1,
              "system:index": "134"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60074688575511, -6.010588864356934]),
            {
              "Class": 1,
              "system:index": "135"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6009614624763, -6.0127228311585235]),
            {
              "Class": 1,
              "system:index": "136"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60147644660717, -6.014856789588715]),
            {
              "Class": 1,
              "system:index": "137"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59559704444652, -6.004869792070147]),
            {
              "Class": 1,
              "system:index": "138"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59173466346508, -6.005894107940526]),
            {
              "Class": 1,
              "system:index": "139"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59589745185619, -6.008796325776308]),
            {
              "Class": 1,
              "system:index": "140"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59190632484203, -6.009095082616401]),
            {
              "Class": 1,
              "system:index": "141"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59169174812084, -6.013832490587157]),
            {
              "Class": 1,
              "system:index": "142"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59160591743236, -6.017161455287208]),
            {
              "Class": 1,
              "system:index": "143"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59589745185619, -6.01127173464407]),
            {
              "Class": 1,
              "system:index": "144"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59598328254467, -6.01536893836598]),
            {
              "Class": 1,
              "system:index": "145"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60636879585033, -6.012850868900433]),
            {
              "Class": 1,
              "system:index": "146"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60533882758861, -6.010759582009059]),
            {
              "Class": 1,
              "system:index": "147"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6055534043098, -6.0077293357232175]),
            {
              "Class": 1,
              "system:index": "148"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60546757362133, -6.004229593672934]),
            {
              "Class": 1,
              "system:index": "149"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6099307694221, -6.004656392688007]),
            {
              "Class": 1,
              "system:index": "150"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60988785407787, -6.0075159374616645]),
            {
              "Class": 1,
              "system:index": "151"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6099307694221, -6.009948672684458]),
            {
              "Class": 1,
              "system:index": "152"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60937286994701, -6.00102859040283]),
            {
              "Class": 1,
              "system:index": "153"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60139061591869, -6.001668792561138]),
            {
              "Class": 1,
              "system:index": "154"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61276318214183, -5.999406741569784]),
            {
              "Class": 1,
              "system:index": "155"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61735512397533, -6.005595349344003]),
            {
              "Class": 1,
              "system:index": "156"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62181831977611, -6.000601788546187]),
            {
              "Class": 1,
              "system:index": "157"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62301994941478, -5.997272722596794]),
            {
              "Class": 1,
              "system:index": "158"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60177685401683, -5.997102000718194]),
            {
              "Class": 1,
              "system:index": "159"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59731365821605, -5.997486124869904]),
            {
              "Class": 1,
              "system:index": "160"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59752823493724, -6.0004737479240315]),
            {
              "Class": 1,
              "system:index": "161"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59392334602123, -5.999620143007707]),
            {
              "Class": 1,
              "system:index": "162"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59049011848217, -6.00017498635537]),
            {
              "Class": 1,
              "system:index": "163"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59139134071117, -6.002778474520593]),
            {
              "Class": 1,
              "system:index": "164"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61696888587718, -6.000815189516289]),
            {
              "Class": 1,
              "system:index": "165"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61890007636791, -6.008326850410609]),
            {
              "Class": 1,
              "system:index": "166"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62126042030101, -6.006320905651253]),
            {
              "Class": 1,
              "system:index": "167"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61353565833812, -6.009351159777377]),
            {
              "Class": 1,
              "system:index": "168"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61263443610912, -6.013149623514366]),
            {
              "Class": 1,
              "system:index": "169"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61096073768383, -6.014686073222436]),
            {
              "Class": 1,
              "system:index": "170"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60860039375072, -6.012594793386475]),
            {
              "Class": 1,
              "system:index": "171"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60684086463695, -6.016009123659167]),
            {
              "Class": 1,
              "system:index": "172"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6049955048347, -6.01289354814104]),
            {
              "Class": 1,
              "system:index": "173"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6031930603767, -6.015070184971205]),
            {
              "Class": 1,
              "system:index": "174"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59448124549633, -6.018527178497802]),
            {
              "Class": 1,
              "system:index": "175"
            }),
        ee.Feature(
            ee.Geometry.Point([106.594781652906, -6.020533078240629]),
            {
              "Class": 1,
              "system:index": "176"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59885861060863, -6.019338075030224]),
            {
              "Class": 1,
              "system:index": "177"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5899751343513, -6.027361617572148]),
            {
              "Class": 1,
              "system:index": "178"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59156300208812, -6.026614750266274]),
            {
              "Class": 1,
              "system:index": "179"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5936658539558, -6.025611812553026]),
            {
              "Class": 1,
              "system:index": "180"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59606911323314, -6.025121012571195]),
            {
              "Class": 1,
              "system:index": "181"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5986011185432, -6.02431012467249]),
            {
              "Class": 1,
              "system:index": "182"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60031773231273, -6.023349861120874]),
            {
              "Class": 1,
              "system:index": "183"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60254933021312, -6.022602988295566]),
            {
              "Class": 1,
              "system:index": "184"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60563923499828, -6.021536025334783]),
            {
              "Class": 1,
              "system:index": "185"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60302139899974, -6.020469060278144]),
            {
              "Class": 1,
              "system:index": "186"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60795666358715, -6.02068245345712]),
            {
              "Class": 1,
              "system:index": "187"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61020971915966, -6.019466111214196]),
            {
              "Class": 1,
              "system:index": "188"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61297775886302, -6.018399142092604]),
            {
              "Class": 1,
              "system:index": "189"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6157457985664, -6.016777345015934]),
            {
              "Class": 1,
              "system:index": "190"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61808468482738, -6.014600715019204]),
            {
              "Class": 1,
              "system:index": "191"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61947943351512, -6.013170963123366]),
            {
              "Class": 1,
              "system:index": "192"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6177628197456, -6.012189340242826]),
            {
              "Class": 1,
              "system:index": "193"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62001596967063, -6.009953843408829]),
            {
              "Class": 1,
              "system:index": "194"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62096010724387, -6.0112555656725615]),
            {
              "Class": 1,
              "system:index": "195"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62347065488181, -6.010679393907151]),
            {
              "Class": 1,
              "system:index": "196"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62540184537254, -6.007478428547184]),
            {
              "Class": 1,
              "system:index": "197"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62739740887962, -6.001823343758094]),
            {
              "Class": 1,
              "system:index": "198"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62904964963279, -5.999134490402672]),
            {
              "Class": 1,
              "system:index": "199"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59092983229269, -6.0288646850574015]),
            {
              "Class": 1,
              "system:index": "200"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59118732435812, -6.030571801737838]),
            {
              "Class": 1,
              "system:index": "201"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59277519209493, -6.027541665935167]),
            {
              "Class": 1,
              "system:index": "202"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59414848311056, -6.028736651089805]),
            {
              "Class": 1,
              "system:index": "203"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59535011274923, -6.026517390850885]),
            {
              "Class": 1,
              "system:index": "204"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59723838789571, -6.027669700184763]),
            {
              "Class": 1,
              "system:index": "205"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59753879530538, -6.025791861496559]),
            {
              "Class": 1,
              "system:index": "206"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60011371595968, -6.026944172370946]),
            {
              "Class": 1,
              "system:index": "207"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59998496992696, -6.024767583110035]),
            {
              "Class": 1,
              "system:index": "208"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60178741438497, -6.023999373051883]),
            {
              "Class": 1,
              "system:index": "209"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60221656782736, -6.025151687731472]),
            {
              "Class": 1,
              "system:index": "210"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60380443556417, -6.023231161906781]),
            {
              "Class": 1,
              "system:index": "211"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60539230330099, -6.024042051416959]),
            {
              "Class": 1,
              "system:index": "212"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60736640913595, -6.024554191536246]),
            {
              "Class": 1,
              "system:index": "213"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60929759962667, -6.023828659558014]),
            {
              "Class": 1,
              "system:index": "214"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60736640913595, -6.021993486109324]),
            {
              "Class": 1,
              "system:index": "215"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61058505995382, -6.020969200557798]),
            {
              "Class": 1,
              "system:index": "216"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61152919752706, -6.02237759269315]),
            {
              "Class": 1,
              "system:index": "217"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61444744093527, -6.021353307865964]),
            {
              "Class": 1,
              "system:index": "218"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61243041975607, -6.019518126053738]),
            {
              "Class": 1,
              "system:index": "219"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6242321394216, -6.008378866315509]),
            {
              "Class": 1,
              "system:index": "220"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62517627699484, -6.0109396358731635]),
            {
              "Class": 1,
              "system:index": "221"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62397464735616, -6.012988242842606]),
            {
              "Class": 1,
              "system:index": "222"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62208637220968, -6.0117932230478965]),
            {
              "Class": 1,
              "system:index": "223"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62174304945577, -6.013116280522048]),
            {
              "Class": 1,
              "system:index": "224"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62268718702902, -6.014823446701063]),
            {
              "Class": 1,
              "system:index": "225"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62079891188253, -6.014482013893981]),
            {
              "Class": 1,
              "system:index": "226"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6196401975881, -6.014994163024225]),
            {
              "Class": 1,
              "system:index": "227"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61921104414571, -6.01631721271126]),
            {
              "Class": 1,
              "system:index": "228"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61727985365499, -6.01657328647235]),
            {
              "Class": 1,
              "system:index": "229"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61607822401632, -6.018920623660933]),
            {
              "Class": 1,
              "system:index": "230"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61466201765646, -6.018707229789995]),
            {
              "Class": 1,
              "system:index": "231"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61745151503195, -6.020286342450444]),
            {
              "Class": 1,
              "system:index": "232"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61843856794943, -6.018109726505594]),
            {
              "Class": 1,
              "system:index": "233"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62036975844015, -6.017213470347326]),
            {
              "Class": 1,
              "system:index": "234"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62152847273458, -6.015933101843467]),
            {
              "Class": 1,
              "system:index": "235"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62551959974874, -6.008464225494867]),
            {
              "Class": 1,
              "system:index": "236"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60908302290548, -6.021438665008674]),
            {
              "Class": 1,
              "system:index": "237"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5969407076477, -6.0173850960186925]),
            {
              "Class": 1,
              "system:index": "238"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59934396692505, -6.014184170115974]),
            {
              "Class": 1,
              "system:index": "239"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5955888743042, -6.012861115239293]),
            {
              "Class": 1,
              "system:index": "240"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60037393518677, -6.01631812281174]),
            {
              "Class": 1,
              "system:index": "241"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59264917322388, -6.01518712892607]),
            {
              "Class": 1,
              "system:index": "242"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59282083460083, -6.0180892971879585]),
            {
              "Class": 1,
              "system:index": "243"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59048194833984, -6.019455018066876]),
            {
              "Class": 1,
              "system:index": "244"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59176940866699, -6.0219517176742885]),
            {
              "Class": 1,
              "system:index": "245"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60541648813476, -6.017065004276502]),
            {
              "Class": 1,
              "system:index": "246"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60608167597046, -6.014077472261364]),
            {
              "Class": 1,
              "system:index": "247"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60432214685669, -6.008909672626675]),
            {
              "Class": 1,
              "system:index": "248"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60213346430054, -6.008781633957392]),
            {
              "Class": 1,
              "system:index": "249"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59833545633545, -6.006370233395024]),
            {
              "Class": 1,
              "system:index": "250"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59578199335327, -6.006348893519367]),
            {
              "Class": 1,
              "system:index": "251"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59307832666626, -6.0032759426952955]),
            {
              "Class": 1,
              "system:index": "252"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59633989282837, -6.003147902701439]),
            {
              "Class": 1,
              "system:index": "253"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59799213358154, -6.001888841159229]),
            {
              "Class": 1,
              "system:index": "254"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59923667856445, -6.00464170075739]),
            {
              "Class": 1,
              "system:index": "255"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6014253611206, -6.006327553642869]),
            {
              "Class": 1,
              "system:index": "256"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6029917711853, -6.00304120268358]),
            {
              "Class": 1,
              "system:index": "257"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60391445108642, -6.005260558751926]),
            {
              "Class": 1,
              "system:index": "258"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60661811777344, -6.005388598249222]),
            {
              "Class": 1,
              "system:index": "259"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61187524744263, -6.006348893519367]),
            {
              "Class": 1,
              "system:index": "260"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61339874216308, -6.00417222180975]),
            {
              "Class": 1,
              "system:index": "261"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61335582681885, -6.001504720109591]),
            {
              "Class": 1,
              "system:index": "262"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61078090616455, -5.999605450941306]),
            {
              "Class": 1,
              "system:index": "263"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61472911783447, -6.005794056458706]),
            {
              "Class": 1,
              "system:index": "264"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61584491678467, -6.004086861957615]),
            {
              "Class": 1,
              "system:index": "265"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61579476686997, -6.007046158817238]),
            {
              "Class": 1,
              "system:index": "266"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61283721670635, -5.996094924775162]),
            {
              "Class": 1,
              "system:index": "267"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61575546011456, -5.996649771709715]),
            {
              "Class": 1,
              "system:index": "268"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61865224585064, -5.998164927771135]),
            {
              "Class": 1,
              "system:index": "269"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61918868765362, -5.99974410003066]),
            {
              "Class": 1,
              "system:index": "270"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62221421942242, -5.998421010069877]),
            {
              "Class": 1,
              "system:index": "271"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62034740194805, -5.996308327509097]),
            {
              "Class": 1,
              "system:index": "272"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62878026709087, -5.995668119056855]),
            {
              "Class": 1,
              "system:index": "273"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62412395224102, -5.99600956365821]),
            {
              "Class": 1,
              "system:index": "274"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61551942572125, -5.9998721408241975]),
            {
              "Class": 1,
              "system:index": "275"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61330928549297, -5.997439360605001]),
            {
              "Class": 1,
              "system:index": "276"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6060565923167, -6.00239027031816]),
            {
              "Class": 1,
              "system:index": "277"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6051553700877, -5.9975887421592935]),
            {
              "Class": 1,
              "system:index": "278"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5937184308482, -6.005399206402359]),
            {
              "Class": 1,
              "system:index": "279"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59916867956646, -6.011587746085253]),
            {
              "Class": 1,
              "system:index": "280"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59176578268536, -6.011566406414143]),
            {
              "Class": 1,
              "system:index": "281"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61044061454794, -6.011587746085253]),
            {
              "Class": 1,
              "system:index": "282"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61481797966024, -6.010776837994823]),
            {
              "Class": 1,
              "system:index": "283"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61713540824911, -6.009773871053436]),
            {
              "Class": 1,
              "system:index": "284"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61331434068148, -6.015686540535647]),
            {
              "Class": 1,
              "system:index": "285"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61475200471347, -6.014064735366833]),
            {
              "Class": 1,
              "system:index": "286"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61597509202426, -6.012592302801149]),
            {
              "Class": 1,
              "system:index": "287"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59690753567945, -6.020464207353958]),
            {
              "Class": 1,
              "system:index": "288"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60175696957837, -6.020165456759888]),
            {
              "Class": 1,
              "system:index": "289"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60089866269361, -6.018757058889551]),
            {
              "Class": 1,
              "system:index": "290"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6055978928877, -6.018650361932454]),
            {
              "Class": 1,
              "system:index": "291"
            }),
        ee.Feature(
            ee.Geometry.Point([106.609975258, -6.015790875679273]),
            {
              "Class": 1,
              "system:index": "292"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62150793320498, -6.007808854635518]),
            {
              "Class": 1,
              "system:index": "293"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62470512635073, -6.005717548371232]),
            {
              "Class": 1,
              "system:index": "294"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62427597290835, -6.002580573918186]),
            {
              "Class": 1,
              "system:index": "295"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63157158142886, -5.997672346366746]),
            {
              "Class": 1,
              "system:index": "296"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62805252320132, -6.000126465666624]),
            {
              "Class": 1,
              "system:index": "297"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62292413956484, -6.009046562722221]),
            {
              "Class": 1,
              "system:index": "298"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61824636704287, -6.010732401967916]),
            {
              "Class": 1,
              "system:index": "299"
            })])

land = ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([106.63115175431739, -6.000348830502721]),
            {
              "Class": 0,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6303792781211, -6.001799956125153]),
            {
              "Class": 0,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62939222520362, -6.003635197702152]),
            {
              "Class": 0,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6287055796958, -6.004531476225544]),
            {
              "Class": 0,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62874849504004, -6.006238669333031]),
            {
              "Class": 0,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6287055796958, -6.0082019347901285]),
            {
              "Class": 0,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62960680192481, -6.009525000996073]),
            {
              "Class": 0,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6301647013999, -6.010976102168868]),
            {
              "Class": 0,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63123758500586, -6.012725954318454]),
            {
              "Class": 0,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63132341569434, -6.014561159061946]),
            {
              "Class": 0,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62784727281104, -6.009396962471544]),
            {
              "Class": 0,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62883432572852, -6.011018781556544]),
            {
              "Class": 0,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62810476487647, -6.012341840914093]),
            {
              "Class": 0,
              "system:index": "12"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62964971726905, -6.012175841731336]),
            {
              "Class": 0,
              "system:index": "13"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62969263261328, -6.0140110483309615]),
            {
              "Class": 0,
              "system:index": "14"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63050802415381, -6.0157608907166775]),
            {
              "Class": 0,
              "system:index": "15"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62561567491065, -6.020967705473684]),
            {
              "Class": 0,
              "system:index": "16"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6226545161582, -6.01853501954212]),
            {
              "Class": 0,
              "system:index": "17"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61690386003028, -6.023315023676696]),
            {
              "Class": 0,
              "system:index": "18"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6171613520957, -6.025662331732748]),
            {
              "Class": 0,
              "system:index": "19"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61711843675147, -6.027412136588986]),
            {
              "Class": 0,
              "system:index": "20"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61677511399756, -6.028649800031657]),
            {
              "Class": 0,
              "system:index": "21"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61600263780127, -6.029418003507002]),
            {
              "Class": 0,
              "system:index": "22"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61441477006446, -6.0276255270391355]),
            {
              "Class": 0,
              "system:index": "23"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61265524095069, -6.026174470323409]),
            {
              "Class": 0,
              "system:index": "24"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61746175950537, -6.031210474051848]),
            {
              "Class": 0,
              "system:index": "25"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6202941722251, -6.0242112697685455]),
            {
              "Class": 0,
              "system:index": "26"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62385614579688, -6.023016274650679]),
            {
              "Class": 0,
              "system:index": "27"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61896379655371, -6.029844782745462]),
            {
              "Class": 0,
              "system:index": "28"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60413301275915, -6.027863769267013]),
            {
              "Class": 0,
              "system:index": "29"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60533464239782, -6.029570889097242]),
            {
              "Class": 0,
              "system:index": "30"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60675084875768, -6.031832814596743]),
            {
              "Class": 0,
              "system:index": "31"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60765207098669, -6.033241178537712]),
            {
              "Class": 0,
              "system:index": "32"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6097978381986, -6.034222763363195]),
            {
              "Class": 0,
              "system:index": "33"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61151445196813, -6.033924020343605]),
            {
              "Class": 0,
              "system:index": "34"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62430322455114, -6.032174236494128]),
            {
              "Class": 0,
              "system:index": "35"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62610566900915, -6.033241178537712]),
            {
              "Class": 0,
              "system:index": "36"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62747896002477, -6.0301257018853835]),
            {
              "Class": 0,
              "system:index": "37"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63044011877722, -6.029698922867949]),
            {
              "Class": 0,
              "system:index": "38"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6271785526151, -6.031619425801703]),
            {
              "Class": 0,
              "system:index": "39"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62636316107458, -6.037807666752716]),
            {
              "Class": 0,
              "system:index": "40"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62207162665075, -6.03691144312898]),
            {
              "Class": 0,
              "system:index": "41"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62394516813474, -6.04099316220703]),
            {
              "Class": 0,
              "system:index": "42"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62634842741208, -6.041804025152299]),
            {
              "Class": 0,
              "system:index": "43"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62965290891843, -6.041035839234429]),
            {
              "Class": 0,
              "system:index": "44"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62656300413327, -6.044535344030841]),
            {
              "Class": 0,
              "system:index": "45"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62484639036374, -6.042572209980065]),
            {
              "Class": 0,
              "system:index": "46"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62699215757566, -6.0480348261941845]),
            {
              "Class": 0,
              "system:index": "47"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6262625967236, -6.046029028211319]),
            {
              "Class": 0,
              "system:index": "48"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63038246977048, -6.047138619504115]),
            {
              "Class": 0,
              "system:index": "49"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62935250150876, -6.044834081196837]),
            {
              "Class": 0,
              "system:index": "50"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61630623686032, -6.049912587776009]),
            {
              "Class": 0,
              "system:index": "51"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61935322630124, -6.04974188244716]),
            {
              "Class": 0,
              "system:index": "52"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62291519987302, -6.049357795260137]),
            {
              "Class": 0,
              "system:index": "53"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6253613744946, -6.049059060592709]),
            {
              "Class": 0,
              "system:index": "54"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6118001257153, -6.04052371468364]),
            {
              "Class": 0,
              "system:index": "55"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61497586118894, -6.03979820407418]),
            {
              "Class": 0,
              "system:index": "56"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61364548551755, -6.040481037615873]),
            {
              "Class": 0,
              "system:index": "57"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59394734251218, -6.035061022676915]),
            {
              "Class": 0,
              "system:index": "58"
            }),
        ee.Feature(
            ee.Geometry.Point([106.595535210249, -6.037194893298327]),
            {
              "Class": 0,
              "system:index": "59"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59806721555906, -6.039072692492706]),
            {
              "Class": 0,
              "system:index": "60"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59776680814939, -6.040907808142143]),
            {
              "Class": 0,
              "system:index": "61"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58956997739988, -6.038176470961978]),
            {
              "Class": 0,
              "system:index": "62"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58819668638425, -6.0396274955542175]),
            {
              "Class": 0,
              "system:index": "63"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59308903562741, -6.039712849820928]),
            {
              "Class": 0,
              "system:index": "64"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59411900388913, -6.040822454063797]),
            {
              "Class": 0,
              "system:index": "65"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59600727903562, -6.0407797770195915]),
            {
              "Class": 0,
              "system:index": "66"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59729473936277, -6.042102763825967]),
            {
              "Class": 0,
              "system:index": "67"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59690850126462, -6.043511101066739]),
            {
              "Class": 0,
              "system:index": "68"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59514897215085, -6.044193929924886]),
            {
              "Class": 0,
              "system:index": "69"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59162991392331, -6.044151253146486]),
            {
              "Class": 0,
              "system:index": "70"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59235947477536, -6.045815645009451]),
            {
              "Class": 0,
              "system:index": "71"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59278862821775, -6.042230794635641]),
            {
              "Class": 0,
              "system:index": "72"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60089962827878, -6.041974732986013]),
            {
              "Class": 0,
              "system:index": "73"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59995549070554, -6.043852515603507]),
            {
              "Class": 0,
              "system:index": "74"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59931176054197, -6.045516908385448]),
            {
              "Class": 0,
              "system:index": "75"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60072796690183, -6.046413117762564]),
            {
              "Class": 0,
              "system:index": "76"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59702705454869, -6.048494131194794]),
            {
              "Class": 0,
              "system:index": "77"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59878658366246, -6.047213836555996]),
            {
              "Class": 0,
              "system:index": "78"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60170482707066, -6.0476728055478475]),
            {
              "Class": 0,
              "system:index": "79"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60397934031529, -6.048825069881204]),
            {
              "Class": 0,
              "system:index": "80"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60586761546178, -6.04822759979262]),
            {
              "Class": 0,
              "system:index": "81"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6057817847733, -6.0503187422141425]),
            {
              "Class": 0,
              "system:index": "82"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61372112345738, -6.049977331759018]),
            {
              "Class": 0,
              "system:index": "83"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6092579276566, -6.048867746290838]),
            {
              "Class": 0,
              "system:index": "84"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6136352927689, -6.035595220638035]),
            {
              "Class": 0,
              "system:index": "85"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61522316050572, -6.037344993442705]),
            {
              "Class": 0,
              "system:index": "86"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6167681128983, -6.034101507675903]),
            {
              "Class": 0,
              "system:index": "87"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61522316050572, -6.032351724399282]),
            {
              "Class": 0,
              "system:index": "88"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62002967906041, -6.0321810135333545]),
            {
              "Class": 0,
              "system:index": "89"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61930011820836, -6.035211123412405]),
            {
              "Class": 0,
              "system:index": "90"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61835598063512, -6.037259638802851]),
            {
              "Class": 0,
              "system:index": "91"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61672519755406, -6.0358086078679625]),
            {
              "Class": 0,
              "system:index": "92"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60805629801793, -6.038497279764393]),
            {
              "Class": 0,
              "system:index": "93"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60998748850865, -6.036278059478023]),
            {
              "Class": 0,
              "system:index": "94"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61178993296666, -6.042338216462791]),
            {
              "Class": 0,
              "system:index": "95"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61466526103062, -6.042167508743702]),
            {
              "Class": 0,
              "system:index": "96"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61882804942174, -6.04225286260997]),
            {
              "Class": 0,
              "system:index": "97"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61681102824254, -6.0378144437215004]),
            {
              "Class": 0,
              "system:index": "98"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61719726634068, -6.041783416178921]),
            {
              "Class": 0,
              "system:index": "99"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61736892771764, -6.039606886498005]),
            {
              "Class": 0,
              "system:index": "100"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62028717112584, -6.039308146447208]),
            {
              "Class": 0,
              "system:index": "101"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62243293833775, -6.034741670878064]),
            {
              "Class": 0,
              "system:index": "102"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62569450449986, -6.030815324671058]),
            {
              "Class": 0,
              "system:index": "103"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62805484843297, -6.032906534342368]),
            {
              "Class": 0,
              "system:index": "104"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62256168437047, -6.038326570834367]),
            {
              "Class": 0,
              "system:index": "105"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62148880076451, -6.040801845053066]),
            {
              "Class": 0,
              "system:index": "106"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62376331400914, -6.0434478153245035]),
            {
              "Class": 0,
              "system:index": "107"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62470745158238, -6.044813472338849]),
            {
              "Class": 0,
              "system:index": "108"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62333416056676, -6.044984179223295]),
            {
              "Class": 0,
              "system:index": "109"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6240637214188, -6.047246040356894]),
            {
              "Class": 0,
              "system:index": "110"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6217033774857, -6.047075334186215]),
            {
              "Class": 0,
              "system:index": "111"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61895679545445, -6.0473740699495355]),
            {
              "Class": 0,
              "system:index": "112"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61659645152135, -6.047160687278285]),
            {
              "Class": 0,
              "system:index": "113"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61492275309605, -6.047203363819275]),
            {
              "Class": 0,
              "system:index": "114"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61234783244176, -6.047331393422018]),
            {
              "Class": 0,
              "system:index": "115"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60908626627965, -6.045709682881296]),
            {
              "Class": 0,
              "system:index": "116"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60599636149449, -6.043063723668266]),
            {
              "Class": 0,
              "system:index": "117"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60951541972203, -6.042508924128037]),
            {
              "Class": 0,
              "system:index": "118"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60732673716588, -6.040759168007221]),
            {
              "Class": 0,
              "system:index": "119"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6034214408402, -6.041655385263436]),
            {
              "Class": 0,
              "system:index": "120"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61204742503209, -6.043874583511718]),
            {
              "Class": 0,
              "system:index": "121"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6066830070023, -6.044728118876436]),
            {
              "Class": 0,
              "system:index": "122"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61200450968785, -6.0451975627531045]),
            {
              "Class": 0,
              "system:index": "123"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61487983775181, -6.043831906708141]),
            {
              "Class": 0,
              "system:index": "124"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61372112345738, -6.037601057281572]),
            {
              "Class": 0,
              "system:index": "125"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61350654673619, -6.042380893384149]),
            {
              "Class": 0,
              "system:index": "126"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61741184306187, -6.044301351362367]),
            {
              "Class": 0,
              "system:index": "127"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61964344096226, -6.044215997819167]),
            {
              "Class": 0,
              "system:index": "128"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62213253092808, -6.044173321042511]),
            {
              "Class": 0,
              "system:index": "129"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62256168437047, -6.041783416178921]),
            {
              "Class": 0,
              "system:index": "130"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61895679545445, -6.039692240767947]),
            {
              "Class": 0,
              "system:index": "131"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62093090128941, -6.0420394779190545]),
            {
              "Class": 0,
              "system:index": "132"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6261880309586, -6.036214043373324]),
            {
              "Class": 0,
              "system:index": "133"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63322614741368, -6.033205277922702]),
            {
              "Class": 0,
              "system:index": "134"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63226055216832, -6.036278059478023]),
            {
              "Class": 0,
              "system:index": "135"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63243221354527, -6.0385186183768615]),
            {
              "Class": 0,
              "system:index": "136"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63402008128209, -6.038582634209216]),
            {
              "Class": 0,
              "system:index": "137"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63009332728429, -6.03713160681786]),
            {
              "Class": 0,
              "system:index": "138"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63281845164342, -6.0418474316253095]),
            {
              "Class": 0,
              "system:index": "139"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63277553629918, -6.040311058822796]),
            {
              "Class": 0,
              "system:index": "140"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63086580348057, -6.040609798320277]),
            {
              "Class": 0,
              "system:index": "141"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62009405207677, -6.037665073222384]),
            {
              "Class": 0,
              "system:index": "142"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59523540444489, -6.0467666588403794]),
            {
              "Class": 0,
              "system:index": "143"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59381919808503, -6.0455503751207935]),
            {
              "Class": 0,
              "system:index": "144"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59347587533112, -6.043288506898058]),
            {
              "Class": 0,
              "system:index": "145"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58877664513703, -6.043907320841885]),
            {
              "Class": 0,
              "system:index": "146"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5877252192032, -6.045379668414993]),
            {
              "Class": 0,
              "system:index": "147"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58789688058015, -6.04826033685936]),
            {
              "Class": 0,
              "system:index": "148"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5890770525467, -6.048793792294877]),
            {
              "Class": 0,
              "system:index": "149"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59163051552888, -6.04877245408756]),
            {
              "Class": 0,
              "system:index": "150"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59276777215119, -6.0481963221717265]),
            {
              "Class": 0,
              "system:index": "151"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58824020333405, -6.046233201405313]),
            {
              "Class": 0,
              "system:index": "152"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58886247582551, -6.047321454014713]),
            {
              "Class": 0,
              "system:index": "153"
            }),
        ee.Feature(
            ee.Geometry.Point([106.58978515572663, -6.046147848166887]),
            {
              "Class": 0,
              "system:index": "154"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59038597054597, -6.044376765430693]),
            {
              "Class": 0,
              "system:index": "155"
            }),
        ee.Feature(
            ee.Geometry.Point([106.591651973201, -6.045934465011887]),
            {
              "Class": 0,
              "system:index": "156"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59117990441437, -6.047001379945122]),
            {
              "Class": 0,
              "system:index": "157"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59768157906647, -6.044568810826911]),
            {
              "Class": 0,
              "system:index": "158"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59731679864045, -6.045507698449389]),
            {
              "Class": 0,
              "system:index": "159"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59845405526276, -6.045080931550235]),
            {
              "Class": 0,
              "system:index": "160"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5979390711319, -6.0474708218490205]),
            {
              "Class": 0,
              "system:index": "161"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60289585536361, -6.040535781567508]),
            {
              "Class": 0,
              "system:index": "162"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59675896113754, -6.037868458427784]),
            {
              "Class": 0,
              "system:index": "163"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59527838176132, -6.040663812747767]),
            {
              "Class": 0,
              "system:index": "164"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5952569240892, -6.041901445929317]),
            {
              "Class": 0,
              "system:index": "165"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59686624949813, -6.041303968194908]),
            {
              "Class": 0,
              "system:index": "166"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60878816927473, -6.033710865431188]),
            {
              "Class": 0,
              "system:index": "167"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61531130159895, -6.028120070645911]),
            {
              "Class": 0,
              "system:index": "168"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61584774340193, -6.030851458505048]),
            {
              "Class": 0,
              "system:index": "169"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61206338912461, -6.0356856661558345]),
            {
              "Class": 0,
              "system:index": "170"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61423061400865, -6.036347166184684]),
            {
              "Class": 0,
              "system:index": "171"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61536787063096, -6.034042581945273]),
            {
              "Class": 0,
              "system:index": "172"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62189100295518, -6.0321007487296345]),
            {
              "Class": 0,
              "system:index": "173"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62461612731431, -6.030799076416933]),
            {
              "Class": 0,
              "system:index": "174"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62525985747789, -6.032250120762389]),
            {
              "Class": 0,
              "system:index": "175"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62871454268907, -6.030692381826526]),
            {
              "Class": 0,
              "system:index": "176"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62944410354112, -6.029262672290079]),
            {
              "Class": 0,
              "system:index": "177"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63006637603257, -6.027576890533164]),
            {
              "Class": 0,
              "system:index": "178"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63060281783555, -6.030500331510893]),
            {
              "Class": 0,
              "system:index": "179"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62712667495225, -6.032655558929544]),
            {
              "Class": 0,
              "system:index": "180"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62639271478506, -6.034264719665479]),
            {
              "Class": 0,
              "system:index": "181"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62199389200063, -6.035523706198493]),
            {
              "Class": 0,
              "system:index": "182"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62536274652334, -6.0410988285709655]),
            {
              "Class": 0,
              "system:index": "183"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62750851373525, -6.041269536627052]),
            {
              "Class": 0,
              "system:index": "184"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62808787088247, -6.039007650516048]),
            {
              "Class": 0,
              "system:index": "185"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62909638147207, -6.039711823620049]),
            {
              "Class": 0,
              "system:index": "186"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63021218042226, -6.03802607435862]),
            {
              "Class": 0,
              "system:index": "187"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63094174127431, -6.042129254112594]),
            {
              "Class": 0,
              "system:index": "188"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61465536813589, -6.040848944413068]),
            {
              "Class": 0,
              "system:index": "189"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61287438135, -6.041297053152331]),
            {
              "Class": 0,
              "system:index": "190"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61032091836782, -6.044241758496041]),
            {
              "Class": 0,
              "system:index": "191"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62559878091665, -6.043580268111232]),
            {
              "Class": 0,
              "system:index": "192"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62523400049062, -6.046530703441205]),
            {
              "Class": 0,
              "system:index": "193"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61858212213369, -6.040833348363525]),
            {
              "Class": 0,
              "system:index": "194"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62755142907949, -6.0432872726981]),
            {
              "Class": 0,
              "system:index": "195"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62276636819692, -6.039937129746592]),
            {
              "Class": 0,
              "system:index": "196"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61047112207265, -6.041942950310021]),
            {
              "Class": 0,
              "system:index": "197"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61328207712026, -6.04848773281905]),
            {
              "Class": 0,
              "system:index": "198"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61819588403554, -6.048573085688201]),
            {
              "Class": 0,
              "system:index": "199"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63058361036886, -6.013137620132359]),
            {
              "Class": 0,
              "system:index": "200"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62897428495992, -6.012817525889807]),
            {
              "Class": 0,
              "system:index": "201"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62452516934717, -6.019649830597599]),
            {
              "Class": 0,
              "system:index": "202"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62712154767358, -6.022872063535475]),
            {
              "Class": 0,
              "system:index": "203"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61690769574487, -6.024344468249195]),
            {
              "Class": 0,
              "system:index": "204"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61494298283318, -6.034812558600031]),
            {
              "Class": 0,
              "system:index": "205"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61725760445763, -6.0367274606202725]),
            {
              "Class": 0,
              "system:index": "206"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61624909386803, -6.040325070856225]),
            {
              "Class": 0,
              "system:index": "207"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61594868645837, -6.041626720284632]),
            {
              "Class": 0,
              "system:index": "208"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61845977772353, -6.023290839150221]),
            {
              "Class": 0,
              "system:index": "209"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6233521269667, -6.021242271008851]),
            {
              "Class": 0,
              "system:index": "210"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62118490208266, -6.022117181264576]),
            {
              "Class": 0,
              "system:index": "211"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6317635344374, -6.0289235697184065]),
            {
              "Class": 0,
              "system:index": "212"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62030163961485, -6.040498150388549]),
            {
              "Class": 0,
              "system:index": "213"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60459814368149, -6.039799924321568]),
            {
              "Class": 0,
              "system:index": "214"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60598502506966, -6.04120329724995]),
            {
              "Class": 0,
              "system:index": "215"
            }),
        ee.Feature(
            ee.Geometry.Point([106.5992347159165, -6.040219181760412]),
            {
              "Class": 0,
              "system:index": "216"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59032978198705, -6.041392800465796]),
            {
              "Class": 0,
              "system:index": "217"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59178890369115, -6.041840908754894]),
            {
              "Class": 0,
              "system:index": "218"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59133829257665, -6.040304535933857]),
            {
              "Class": 0,
              "system:index": "219"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59099496982275, -6.038618788517352]),
            {
              "Class": 0,
              "system:index": "220"
            }),
        ee.Feature(
            ee.Geometry.Point([106.59921325824438, -6.0493306637943975]),
            {
              "Class": 0,
              "system:index": "221"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60268940112768, -6.047068811378802]),
            {
              "Class": 0,
              "system:index": "222"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60110153339086, -6.046834090303319]),
            {
              "Class": 0,
              "system:index": "223"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60290397784887, -6.048071709372022]),
            {
              "Class": 0,
              "system:index": "224"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60444893024145, -6.046428662751202]),
            {
              "Class": 0,
              "system:index": "225"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60195984027563, -6.042289016673038]),
            {
              "Class": 0,
              "system:index": "226"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62100863409701, -6.032680695933283]),
            {
              "Class": 0,
              "system:index": "227"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61963534308138, -6.033427554886575]),
            {
              "Class": 0,
              "system:index": "228"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62624430609408, -6.031869819333384]),
            {
              "Class": 0,
              "system:index": "229"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62851881933871, -6.031549736131186]),
            {
              "Class": 0,
              "system:index": "230"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63032042929277, -6.03575555515726]),
            {
              "Class": 0,
              "system:index": "231"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62910752235436, -6.041826932646772]),
            {
              "Class": 0,
              "system:index": "232"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62792735038781, -6.045817214010204]),
            {
              "Class": 0,
              "system:index": "233"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6247730725863, -6.043576685316037]),
            {
              "Class": 0,
              "system:index": "234"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6174989217379, -6.040482606632083]),
            {
              "Class": 0,
              "system:index": "235"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60892914917976, -6.037026774269984]),
            {
              "Class": 0,
              "system:index": "236"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62023734238655, -6.030917259780257]),
            {
              "Class": 0,
              "system:index": "237"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63045912000456, -6.01734180435162]),
            {
              "Class": 0,
              "system:index": "238"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63041620466032, -6.019497084012459]),
            {
              "Class": 0,
              "system:index": "239"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63097410413542, -6.020606729613111]),
            {
              "Class": 0,
              "system:index": "240"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63189678403654, -6.022271193764179]),
            {
              "Class": 0,
              "system:index": "241"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63311987134733, -6.023167441578872]),
            {
              "Class": 0,
              "system:index": "242"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63318424436369, -6.024085027094765]),
            {
              "Class": 0,
              "system:index": "243"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63202553006926, -6.026112245410927]),
            {
              "Class": 0,
              "system:index": "244"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63142471524992, -6.026667061735745]),
            {
              "Class": 0,
              "system:index": "245"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62637846622364, -6.0270195138291305]),
            {
              "Class": 0,
              "system:index": "246"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62723677310841, -6.026528715120246]),
            {
              "Class": 0,
              "system:index": "247"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62685053501026, -6.013298321926212]),
            {
              "Class": 0,
              "system:index": "248"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62189381275074, -6.02031900617816]),
            {
              "Class": 0,
              "system:index": "249"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62972586307423, -6.021642042902018]),
            {
              "Class": 0,
              "system:index": "250"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62616388950245, -6.0179930144493845]),
            {
              "Class": 0,
              "system:index": "251"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62627117786305, -6.015667012765968]),
            {
              "Class": 0,
              "system:index": "252"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62760155353443, -6.016968721356496]),
            {
              "Class": 0,
              "system:index": "253"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62798779163258, -6.015090845673493]),
            {
              "Class": 0,
              "system:index": "254"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6283525720586, -6.01839846326538]),
            {
              "Class": 0,
              "system:index": "255"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62815945300953, -6.020681774632747]),
            {
              "Class": 0,
              "system:index": "256"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62955420169727, -6.01867587543887]),
            {
              "Class": 0,
              "system:index": "257"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62597077045338, -6.019337396204171]),
            {
              "Class": 0,
              "system:index": "258"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63051979694264, -6.024138732456281]),
            {
              "Class": 0,
              "system:index": "259"
            }),
        ee.Feature(
            ee.Geometry.Point([106.63150684986012, -6.024373463362463]),
            {
              "Class": 0,
              "system:index": "260"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61366849352541, -6.024045937784101]),
            {
              "Class": 0,
              "system:index": "261"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6144409697217, -6.02513423495231]),
            {
              "Class": 0,
              "system:index": "262"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61497741152468, -6.023704510771965]),
            {
              "Class": 0,
              "system:index": "263"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60586198922158, -6.026862702440758]),
            {
              "Class": 0,
              "system:index": "264"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60684904213906, -6.028015011041367]),
            {
              "Class": 0,
              "system:index": "265"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60221418496133, -6.028932588362027]),
            {
              "Class": 0,
              "system:index": "266"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60463890191079, -6.03209074960066]),
            {
              "Class": 0,
              "system:index": "267"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60517534371377, -6.033627145714794]),
            {
              "Class": 0,
              "system:index": "268"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60504659768105, -6.035035504994063]),
            {
              "Class": 0,
              "system:index": "269"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6063340580082, -6.035718344539357]),
            {
              "Class": 0,
              "system:index": "270"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60635551568032, -6.036891972997364]),
            {
              "Class": 0,
              "system:index": "271"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60592636223794, -6.038001583019542]),
            {
              "Class": 0,
              "system:index": "272"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60427412148476, -6.037638826146972]),
            {
              "Class": 0,
              "system:index": "273"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60916647072793, -6.026862702440758]),
            {
              "Class": 0,
              "system:index": "274"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60850128289223, -6.028057689090703]),
            {
              "Class": 0,
              "system:index": "275"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6083510791874, -6.029828825176962]),
            {
              "Class": 0,
              "system:index": "276"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60972437020303, -6.030639704828742]),
            {
              "Class": 0,
              "system:index": "277"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6090162670231, -6.031749327650707]),
            {
              "Class": 0,
              "system:index": "278"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61180576439858, -6.031727988771695]),
            {
              "Class": 0,
              "system:index": "279"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61425193902016, -6.032069410735083]),
            {
              "Class": 0,
              "system:index": "280"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61337217446328, -6.030170248335996]),
            {
              "Class": 0,
              "system:index": "281"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61298593636513, -6.031471922158075]),
            {
              "Class": 0,
              "system:index": "282"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61191305275918, -6.029892842035643]),
            {
              "Class": 0,
              "system:index": "283"
            }),
        ee.Feature(
            ee.Geometry.Point([106.60923084374429, -6.028932588362027]),
            {
              "Class": 0,
              "system:index": "284"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6213544284916, -6.0271614493499515]),
            {
              "Class": 0,
              "system:index": "285"
            }),
        ee.Feature(
            ee.Geometry.Point([106.61993822213174, -6.026115834448279]),
            {
              "Class": 0,
              "system:index": "286"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62219127770425, -6.02588110429552]),
            {
              "Class": 0,
              "system:index": "287"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62229856606484, -6.023981920235673]),
            {
              "Class": 0,
              "system:index": "288"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62476619835854, -6.025987799832091]),
            {
              "Class": 0,
              "system:index": "289"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6238220607853, -6.024536738737885]),
            {
              "Class": 0,
              "system:index": "290"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62519535180093, -6.024152633681362]),
            {
              "Class": 0,
              "system:index": "291"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62667593117715, -6.025411643685469]),
            {
              "Class": 0,
              "system:index": "292"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62553867455483, -6.025091556673095]),
            {
              "Class": 0,
              "system:index": "293"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6254957592106, -6.027673586525927]),
            {
              "Class": 0,
              "system:index": "294"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62489494439126, -6.028825893404481]),
            {
              "Class": 0,
              "system:index": "295"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62716945763589, -6.027950993961089]),
            {
              "Class": 0,
              "system:index": "296"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6294225132084, -6.024707452009115]),
            {
              "Class": 0,
              "system:index": "297"
            }),
        ee.Feature(
            ee.Geometry.Point([106.62875634204856, -6.0239499813082515]),
            {
              "Class": 0,
              "system:index": "298"
            }),
        ee.Feature(
            ee.Geometry.Point([106.6272328473281, -6.01880721329718]),
            {
              "Class": 0,
              "system:index": "299"
            })])
training_fc = water.merge(land)

# -------------------------
# 5) Pemilihan Fitur dan Pembuatan Training Set serta Test Set
# -------------------------
bands = ['B4', 'B5', 'B6', 'B7', 'B8', 'B10', 'B11', 'NDWI']
input_image = image.select(bands)

sampled = input_image.sampleRegions(
    collection=training_fc,
    properties=['Class'],
    scale=30,
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
def init_predict_landsat (startDate, endDate):
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
    image = (l8
         .filterDate(startDate, endDate)
         .filter(ee.Filter.lt('CLOUD_COVER', 20))
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
    # out_path = f"prediction/prediction_{startDate}_{endDate}.png"
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

    # download prediksi dalam bentuk GeoTIFF
    response = requests.get(url)
    filename = f"../web_app/static/assets/custom_model/raw_data.tif"
    with open(filename, "wb") as f:
        f.write(response.content)
    
    return filename