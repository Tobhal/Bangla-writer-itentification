import unittest

import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from numpy import ndarray

from Util import h_from_image, division
import math
from numba import njit


@njit
def gradient(img, print_values=False) -> ndarray:
    img = 255.0 - img

    if print_values:
        print(img.size)
        print(img.shape)
        print(img[0])

    img = img / 255.0

    """
    if img_show:
        plt.imshow(img, cmap='Greys')
        plt.show()
    """

    d = np.zeros((9, 9, 16))
    nc1 = np.zeros((11, 11, 16))
    nc2 = np.zeros((5, 5, 16))
    appe = np.zeros((400,))

    h1, h2, h3, h4 = h_from_image(img)
    ro = h4 - h3 + 1
    co = h2 - h1 + 1

    if print_values:
        print(h1, h2, h3, h4)

    counter = 0
    for i in range(h3, h4 + 1):
        for j in range(h1, h2 + 1):
            if img[i][j] == 1:
                counter = counter + 1
    if print_values:
        print('counter is:', counter)

    # 1: Mean filter?
    for k in range(1, 6):
        for i in range(h3, h4 + 1):
            for j in range(h1, h2 + 1):
                s = img[i][j]
                if h3 <= i <= h4 and h1 <= j + 1 <= h2:
                    s = s + img[i][j + 1]
                if h3 <= i + 1 <= h4 and h1 <= j <= h2:
                    s = s + img[i + 1][j]
                if h3 <= i + 1 <= h4 and h1 <= j + 1 <= h2:
                    s = s + img[i + 1][j + 1]
                img[i][j] = s

    total = 0
    for i in range(h3, h4 + 1):
        for j in range(h1, h2 + 1):
            total = total + img[i][j]

    count = ro * co
    mean = total / count

    if print_values:
        print('count is:', ro * co, ' total is:', total)
        print('mean is: ', mean)

    # 2: Nomilize graysale image
    for i in range(h3, h4 + 1):
        for j in range(h1, h2 + 1):
            img[i][j] = img[i][j] - mean

    maxi = img[h3][h1]
    for i in range(h3, h4 + 1):
        for j in range(h1, h2 + 1):
            if maxi < img[i][j]:
                maxi = img[i][j]

    if print_values:
        print('maxx is :', maxi)

    for i in range(h3, h4 + 1):
        for j in range(h1, h2 + 1):
            img[i][j] = img[i][j] / maxi

    # 3: Segment nomalized image into 9x9 blocks
    r11, c11 = np.zeros((20,)), np.zeros(20, )
    r11 = division(ro, r11, 9)
    c11 = division(co, c11, 9)
    img1 = []

    for i in range(h3, h4 + 1):
        img1_row = []
        for j in range(h1, h2 + 1):
            img1_row.append(img[i][j])
        img1.append(img1_row)
    img1 = np.asarray(img1)

    if print_values:
        print(' shape of img1 is:', img1.shape)

    co = co - 1

    if print_values:
        print(ro, co, ' ro and co is')

    # 4: 
    for i in range(ro):
        img1[i][co - 1] = (img1[i][co - 1] + img1[i][co - 2]) / 2
    for j in range(co):
        img1[ro - 1][j] = (img1[ro - 1][j] + img1[ro - 2][j]) / 2

    img1[ro - 1][co - 1] = (img1[ro - 1][co - 1] + img1[ro - 2][co - 2]) / 2

    for m in range(0, 9):
        for n in range(0, 9):
            s1 = 0
            y, z = 0, 0
            for i in range(0, m):
                y = y + r11[i]

            z = y + r11[m]
            for j in range(0, n):
                s1 = s1 + c11[j]

            t = s1 + c11[n]
            y, z, s1, t = int(y), int(z), int(s1), int(t)
            for i in range(y, z - 1):
                for j in range(s1, t - 1):
                    delu = img1[i + 1][j] - img1[i][j]
                    delv = img1[i][j + 1] - img1[i][j]
                    if delu == 0:
                        continue
                    else:
                        fxy = sqrt((delu * delu) + (delv * delv))
                        angxy = (math.atan2(delv, delu) * 180) / 3.142857
                        angxy = angxy + 180
                        rxy = angxy / 22.5
                        rxy = int(rxy)
                        d[m][n][rxy] = d[m][n][rxy] + fxy

    i, j = 0, 0
    while i < 9:
        while j < 9:
            for k in range(0, 16, 1):
                d[i][j][k] = d[i][j][k] * 2
            j += 2
        i += 2

    for i in range(1, 10, 1):
        for j in range(1, 10, 1):
            for k in range(0, 16, 1):
                nc1[i][j][k] = d[i - 1][j - 1][k]

    for i in range(1, 10, 2):
        for j in range(1, 10, 2):
            for k in range(0, 16, 1):
                nc2[int(i / 2)][int(j / 2)][k] = nc1[i][j][k] + nc1[i - 1][j - 1][k] + nc1[i - 1][j][k] + nc1[i - 1][j + 1][k] + nc1[i][j - 1][k] + nc1[i][j + 1][k] + nc1[i + 1][j - 1][k] + nc1[i + 1][j][k] + nc1[i + 1][j + 1][k]

    h11 = 0
    for i in range(0, 5):
        for j in range(0, 5):
            for k in range(0, 16):
                appe[h11] = nc2[i][j][k] / counter
                h11 = h11 + 1

    return appe

"""
class MyTestCase(unittest.TestCase):
    def test_something(self):
        correct_value = [
            0.004077128824310569, 0.0029490547001662743, 0.0028303714170429426, 0.0029422509281727383,
            0.0014550191530480529, 0.0011805668484631197, 0.0018295019982095598, 0.0043587944103192335,
            0.004454859228041683, 0.002745195676053644, 0.0026906627439745065, 0.0023691441849112726,
            0.0021649724817343497, 0.0016634242810131677, 0.0021436387943297836, 0.003672046752403197,
            0.004944928207503524, 0.0037134614588568505, 0.0035212725474738182, 0.004026751766917179,
            0.002181993406019575, 0.0015091657704769943, 0.002175574350958824, 0.006178280272368041,
            0.005373501017094024, 0.0034148903867015617, 0.0029337629843489963, 0.0034483250688015846,
            0.0026819064529901696, 0.001955973696493071, 0.0023906582171482507, 0.005116511548285561,
            0.005421472893339558, 0.0038846705432307024, 0.003028158184684204, 0.0033826732065764405,
            0.0027135248414103283, 0.0017887692375899626, 0.002189935273241184, 0.005544184333902538,
            0.00610006514471797, 0.003455672924862924, 0.002640794128450522, 0.0030944236315585474,
            0.003224528038265914, 0.0023309515569003136, 0.0023486992495514018, 0.004983189654118158,
            0.0051613841611414185, 0.00284522228041594, 0.0025439471135236774, 0.0026932811036875703,
            0.0035691943454627703, 0.001974126001100227, 0.0023605848941093996, 0.005270333253303571,
            0.0052053437725016905, 0.0023301561237549753, 0.0021527439432493254, 0.0028928690002842936,
            0.004040130191822544, 0.0025493413087050525, 0.002455877090984058, 0.004166017280789138,
            0.0031860017118872826, 0.00190835161377071, 0.002483210025001359, 0.0027108475082431634,
            0.003460429499247947, 0.0009850271421242215, 0.0012185145445187918, 0.004439829931573072,
            0.003658394820182841, 0.0014131643547998448, 0.0015680013964195624, 0.0026883750938758717,
            0.0032862235328947405, 0.0015946919382754492, 0.0015770589061609539, 0.0031546894419287987,
            0.003137573363510818, 0.002107414725123788, 0.002158140874071368, 0.0020720483369600987,
            0.001822919277300198, 0.0009693262899770324, 0.0014183574643892468, 0.0035292379137146514,
            0.003463699187918303, 0.0020898053261718414, 0.00167491652964709, 0.001953240658145016,
            0.0021980681454023404, 0.0012760155727890043, 0.0015599711582162542, 0.003069010317669274,
            0.004150391168058989, 0.003010911564317956, 0.0034341239665713133, 0.0037241052153208255,
            0.0020450394832415403, 0.0013913344661561067, 0.0017169497826077333, 0.004128494712885737,
            0.004389848424849356, 0.002962029112744224, 0.002581362738062338, 0.0032218600679449613,
            0.002836615118507097, 0.001709393681560946, 0.0017959248251522772, 0.003262799475295218,
            0.004310836695499219, 0.0027053624643899204, 0.0026399691220827474, 0.003256426920139023,
            0.002718324236598228, 0.001547180287251844, 0.0017171055217488742, 0.0041445758245661855,
            0.004806665033370265, 0.002483658288686987, 0.002280223174753539, 0.0027423716292621, 0.0034123626186599587,
            0.001978715669751584, 0.0018239032753411427, 0.0036151575915677366, 0.003708290009088253,
            0.0025178508242362533, 0.002251887662980115, 0.0026664859415438903, 0.00354281580823697,
            0.0016139195731993624, 0.001864368329463877, 0.004077821977492102, 0.0038355972515090276,
            0.002075105085738203, 0.0019525542345328706, 0.002627065709358341, 0.003631562408677471,
            0.0022983043016357443, 0.0019858430273407245, 0.0031037952771178438, 0.0032125555068867762,
            0.0016332368648153083, 0.0015206880849416803, 0.001697732795857246, 0.0024799173789252006,
            0.0009735450898891027, 0.001288596870407744, 0.003335144570127515, 0.0033877400091000243,
            0.001323615308481683, 0.0010861607775494908, 0.0017485272977943783, 0.002359686396314252,
            0.0014716779709492058, 0.0015334807833591855, 0.002542236588708242, 0.003367550440780783,
            0.002510269429463687, 0.002359221100886166, 0.0027080310094868927, 0.002035477639821295,
            0.0011761468536396532, 0.0015800711862782786, 0.0038562211122873473, 0.0036405546006131103,
            0.0020402079263940557, 0.0023889391366247134, 0.0024979608582174555, 0.002169090512461793,
            0.0015716542064612888, 0.0019256742611691, 0.003595963932277877, 0.005301566532031829, 0.004184253271749815,
            0.0032824364896939526, 0.003547727798095499, 0.002337063531636688, 0.0015234990626639966,
            0.0021630890276931125, 0.00549248078173843, 0.0058208969960536405, 0.003736824496183787,
            0.0031394146017441377, 0.0031553399925237095, 0.0028124302781133157, 0.0019060465377521954,
            0.002233430794892904, 0.004637332550218693, 0.005265840662749845, 0.003945481239325025,
            0.0024941637158903056, 0.0028424668058563848, 0.0025328594362171923, 0.0015487744223432892,
            0.0018594764748669284, 0.004429658608397333, 0.00663532058076311, 0.003435800227163159,
            0.0024038252781711845, 0.002403303880113253, 0.003045858399932658, 0.0018512343272987048,
            0.0019672378713910792, 0.0037473483231272356, 0.004245120693398577, 0.0027000260633822173,
            0.002287886159593655, 0.0030749565728546775, 0.003287505004873634, 0.0015109211970013945,
            0.0019641218734883542, 0.004711375029262146, 0.0054412694522215175, 0.0022424485380856925,
            0.0022729031305471787, 0.002956840495334693, 0.003376667121475358, 0.0021722729757538517,
            0.0020047233341256064, 0.00343985712640793, 0.003646625805537524, 0.002237565484082655,
            0.002192937939335296, 0.0026982185126311017, 0.002553482208661975, 0.0013220086195315135,
            0.0016935103337619196, 0.003896097660931736, 0.004492202701633464, 0.0019399350851944366,
            0.001869054692621197, 0.0025974683521127573, 0.002858901595718423, 0.0016958300694522401,
            0.001762407666596793, 0.002990435540308728, 0.002503972331028882, 0.0015443386534061375,
            0.0019215704197519537, 0.002345795522453855, 0.0013717722777607114, 0.0010729729319598437,
            0.001361491485397306, 0.0032292398000772224, 0.0026089564409599504, 0.0013738714723882596,
            0.0017360431544149243, 0.001952373430179502, 0.0017771876100867803, 0.0013337795266389883,
            0.0014987380345694455, 0.00258500863495993, 0.003915424653545754, 0.002690810510369244,
            0.0027700881663373497, 0.003359969013940916, 0.0025673305940388494, 0.0016650246849030002,
            0.001955340489323202, 0.00426668915733842, 0.0040164387704464816, 0.0026294495038814447,
            0.002447906744560529, 0.002897913800911256, 0.002913848210986933, 0.0020235635323476225,
            0.0021182795261386377, 0.0036263962473177423, 0.004137740709256411, 0.002757754461317521,
            0.002319160610830984, 0.0023429509397946548, 0.0021965697512481394, 0.001409689800551991,
            0.0015548626389319856, 0.003556503450372202, 0.004665953738621671, 0.002587533079372976,
            0.0016818218517767245, 0.0024066586046190366, 0.0027089099030397734, 0.0015917316595191928,
            0.001655638344676136, 0.0031869568818109276, 0.004468584941907731, 0.0030148761425622195,
            0.0023211588984951587, 0.0029913466090665804, 0.0022694771652368, 0.0014445222357853619,
            0.001832833020635789, 0.004576690877695872, 0.005129518115473797, 0.002250385679054296,
            0.0018253912996837125, 0.0027728502331420906, 0.003071697584482982, 0.001809970126237425,
            0.0020752476171502546, 0.0036890954200179803, 0.0025258747607356828, 0.0019014566218399666,
            0.0018826041988789105, 0.002804221605220831, 0.001750081686269969, 0.0010430547068298727,
            0.0012465576561412459, 0.0027490335668019323, 0.0027510602415653927, 0.0015339785270833558,
            0.001871866434443142, 0.002163405805829991, 0.0021449577333155522, 0.0012899826207570066,
            0.0013416965094218125, 0.0022974067028282484, 0.0023617219248989537, 0.0016678437058437428,
            0.0016962285800598568, 0.001896999451396727, 0.0017950822751203418, 0.0011132205897333115,
            0.0013486037371447308, 0.0027669906987439316, 0.002455123976639927, 0.0016182983261247364,
            0.0015474048338278532, 0.001578833174876335, 0.00201638956053904, 0.0013939072617868017,
            0.0014130409897666412, 0.002362426640308279, 0.004170192105488274, 0.0027996013637855985,
            0.00256438368131173, 0.002861839435788146, 0.0024526317409846424, 0.0015325335841143337,
            0.001930142076642979, 0.004109453852083442, 0.004337069284132057, 0.0028185624386634516,
            0.0022999611073579074, 0.002393544452442593, 0.0027148997534682607, 0.0018853816653821842,
            0.0019769418085103514, 0.0036613757763318835, 0.0045473511075400435, 0.003295187058846744,
            0.0024950370357720267, 0.0028775875386815196, 0.0020167737326044167, 0.0013650368126249277,
            0.0018069709016649885, 0.004034037006643483, 0.004982825806485098, 0.0030782037975620132,
            0.002297076618820858, 0.002489582719818407, 0.002603423088308081, 0.0016658064249134756,
            0.001787699246180791, 0.0037471685709046785, 0.0038711203066211946, 0.0029106804275164976,
            0.0024647067061786265, 0.002905292477602463, 0.0018738716685519304, 0.0010678769573548887,
            0.001433016649778589, 0.004333840132013282, 0.004170635430207997, 0.002346612295067486,
            0.0021911410139412103, 0.002665028456204142, 0.0025115720694367815, 0.0013624369484352963,
            0.001448875865182801, 0.0035717667611368346, 0.0021715607084902126, 0.0017570646867798507,
            0.001635108840893464, 0.002237269336669336, 0.001420276723128006, 0.0007487384717596092,
            0.0008532139948162065, 0.0028455845925441335, 0.0024487769598409556, 0.0013922425087025622,
            0.0015954160185623085, 0.0017150416671078874, 0.0017139418555728583, 0.0009299640798456006,
            0.0009550272429228192, 0.002403796845193213
        ]

        app = gradient('Bangla-writer-test/test_001.tif')

        for _i in range(len(app)):
            self.assertEqual(app[_i], correct_value[_i])
"""

if __name__ == '__main__':
    gradient('Bangla-writer-test/test_001.tif')

    # unittest.main()
