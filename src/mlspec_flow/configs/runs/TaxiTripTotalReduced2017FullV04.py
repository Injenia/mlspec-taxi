from configs.runs.TaxiTripTotalReduced2017FullV02 import TaxiTripTotalReduced2017FullV02Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

company_historical_data={'1408 - 89599 Donald Barnes': {'avg_trip_total': 23.571500593119804,
  'avg_trip_seconds': 1312.7046263345194,
  'avg_total_per_second': 0.018365944275541154,
  'avg_total_per_mile': 4.363848671003216,
  'category': 'normal'},
 '5129 - 98755 Mengisti Taxi': {'avg_trip_total': 17.09016167505963,
  'avg_trip_seconds': 954.6726742645109,
  'avg_total_per_second': 0.019106167049067504,
  'avg_total_per_mile': 5.945213735178227,
  'category': 'normal'},
 'Globe Taxi': {'avg_trip_total': 16.820623633162107,
  'avg_trip_seconds': 855.4685716054613,
  'avg_total_per_second': 0.021594368162708443,
  'avg_total_per_mile': 6.964682692805985,
  'category': 'normal'},
 '5 Star Taxi': {'avg_trip_total': 14.188513576742128,
  'avg_trip_seconds': 802.0251452304021,
  'avg_total_per_second': 0.028272282414084155,
  'avg_total_per_mile': 6.224554996781883,
  'category': 'normal'},
 '2809 - 95474 C&D Cab Co Inc.': {'avg_trip_total': 13.91871657754011,
  'avg_trip_seconds': 803.4224598930482,
  'avg_total_per_second': 0.020480670295918147,
  'avg_total_per_mile': 6.565762128800042,
  'category': 'normal'},
 'Blue Ribbon Taxi Association Inc.': {'avg_trip_total': 19.410838863500633,
  'avg_trip_seconds': 1106.5291950400983,
  'avg_total_per_second': 0.018071946209287607,
  'avg_total_per_mile': 74.87875537461157,
  'category': 'luxury'},
 '0118 - Godfray S.Awir': {'avg_trip_total': 8.7625,
  'avg_trip_seconds': 630.0,
  'avg_total_per_second': 0.01988897670021291,
  'avg_total_per_mile': 12.53106794579534,
  'category': 'expensive'},
 '303 Taxi': {'avg_trip_total': 10.28628003717811,
  'avg_trip_seconds': 1096.7174705862321,
  'avg_total_per_second': 0.0190853824499671,
  'avg_total_per_mile': 2.8383038572981913,
  'category': 'normal'},
 '3152 - Crystal Abernathy': {'avg_trip_total': 10.037920380273324,
  'avg_trip_seconds': 653.3333333333334,
  'avg_total_per_second': 0.0180341763785954,
  'avg_total_per_mile': 4.190393669832172,
  'category': 'normal'},
 '4615 - Tyrone Henderson': {'avg_trip_total': 12.315231099964171,
  'avg_trip_seconds': 797.9505553565032,
  'avg_total_per_second': 0.01870671463687797,
  'avg_total_per_mile': 17.018163558062934,
  'category': 'expensive'},
 '2241 - 44667 Manuel Alonso': {'avg_trip_total': 35.07450951137637,
  'avg_trip_seconds': 1862.6855650876541,
  'avg_total_per_second': 0.02064160287864323,
  'avg_total_per_mile': 4.643356589736362,
  'category': 'normal'},
 '2823 - Seung Lee': {'avg_trip_total': 20.41444723618091,
  'avg_trip_seconds': 1314.2713567839196,
  'avg_total_per_second': 0.016778642916438706,
  'avg_total_per_mile': 4.987310446913645,
  'category': 'normal'},
 'Peace Taxi Assoc': {'avg_trip_total': 16.563558737151247,
  'avg_trip_seconds': 874.5859030837008,
  'avg_total_per_second': 0.02108720649040608,
  'avg_total_per_mile': 6.879510706214461,
  'category': 'normal'},
 'Metro Group': {'avg_trip_total': 13.462601210474892,
  'avg_trip_seconds': 737.4935663735774,
  'avg_total_per_second': 0.055483367920598665,
  'avg_total_per_mile': 4.682213670393517,
  'category': 'normal'},
 'Blue Diamond': {'avg_trip_total': 14.781845379423638,
  'avg_trip_seconds': 811.3737746701888,
  'avg_total_per_second': 0.04362729980650034,
  'avg_total_per_mile': 7.027984910634103,
  'category': 'normal'},
 '2733 - 74600 Benny Jona': {'avg_trip_total': 18.49344067589078,
  'avg_trip_seconds': 1000.9134321048122,
  'avg_total_per_second': 0.019064490131026915,
  'avg_total_per_mile': 6.787550650502189,
  'category': 'normal'},
 '585 - Valley Cab Co': {'avg_trip_total': 12.930393559928444,
  'avg_trip_seconds': 814.5617173524151,
  'avg_total_per_second': 0.018036930323312283,
  'avg_total_per_mile': 5.2487545505250734,
  'category': 'normal'},
 '1408 - Donald Barnes': {'avg_trip_total': 19.902994923857868,
  'avg_trip_seconds': 1239.5939086294416,
  'avg_total_per_second': 0.017186939977141748,
  'avg_total_per_mile': 5.162509609797394,
  'category': 'normal'},
 'Chicago Star Taxicab': {'avg_trip_total': 22.06215064420218,
  'avg_trip_seconds': 1119.7819623389494,
  'avg_total_per_second': 0.02052760209602249,
  'avg_total_per_mile': 6.40923425278995,
  'category': 'normal'},
 'Taxi Affiliation Services': {'avg_trip_total': 16.423301506443064,
  'avg_trip_seconds': 908.1361861747721,
  'avg_total_per_second': 0.019827450357947224,
  'avg_total_per_mile': 28.030299064821378,
  'category': 'expensive'},
 '3620 - David K. Cab Corp.': {'avg_trip_total': 17.775805340223943,
  'avg_trip_seconds': 995.3488372093025,
  'avg_total_per_second': 0.018974660694440953,
  'avg_total_per_mile': 5.532414179315814,
  'category': 'normal'},
 '5864 - Thomas Owusu': {'avg_trip_total': 13.417022900763358,
  'avg_trip_seconds': 813.2388222464558,
  'avg_total_per_second': 0.018041269049272203,
  'avg_total_per_mile': 5.093301558879226,
  'category': 'normal'},
 '5129 - 87128': {'avg_trip_total': 16.264419331570082,
  'avg_trip_seconds': 808.6078384226978,
  'avg_total_per_second': 0.021975010191719078,
  'avg_total_per_mile': 6.903562812463477,
  'category': 'normal'},
 '5724 - 72965 KYVI Cab Inc': {'avg_trip_total': 11.82902674265673,
  'avg_trip_seconds': 732.573432704954,
  'avg_total_per_second': 0.017826895689057782,
  'avg_total_per_mile': 6.328042284349883,
  'category': 'normal'},
 '2092 - Sbeih company': {'avg_trip_total': 13.666427104722793,
  'avg_trip_seconds': 875.7289527720739,
  'avg_total_per_second': 0.01650291400588236,
  'avg_total_per_mile': 5.517531865963469,
  'category': 'normal'},
 'Sun Taxi': {'avg_trip_total': 17.340481220099726,
  'avg_trip_seconds': 906.6406364688186,
  'avg_total_per_second': 0.02192403933707156,
  'avg_total_per_mile': 6.855001651351244,
  'category': 'normal'},
 'Patriot Taxi Dba Peace Taxi Associat': {'avg_trip_total': 15.914697173311682,
  'avg_trip_seconds': 853.7087108150978,
  'avg_total_per_second': 0.021249672917827058,
  'avg_total_per_mile': 8.222186444315215,
  'category': 'normal'},
 'Checker Taxi Affiliation': {'avg_trip_total': 17.43301947752239,
  'avg_trip_seconds': 852.2865285870869,
  'avg_total_per_second': 0.02370656956755982,
  'avg_total_per_mile': 10.002456318167647,
  'category': 'expensive'},
 '585 - 88805 Valley Cab Co': {'avg_trip_total': 13.756803898170247,
  'avg_trip_seconds': 755.2207637231503,
  'avg_total_per_second': 0.020114683682391478,
  'avg_total_per_mile': 6.055666367007106,
  'category': 'normal'},
 '3201 - CD Cab Co Inc': {'avg_trip_total': 13.344463629684052,
  'avg_trip_seconds': 832.1969140337986,
  'avg_total_per_second': 0.020692561221238286,
  'avg_total_per_mile': 7.25736796995552,
  'category': 'normal'},
 '5874 - Sergey Cab Corp.': {'avg_trip_total': 17.434563793555142,
  'avg_trip_seconds': 840.37725962798,
  'avg_total_per_second': 0.021557145999543708,
  'avg_total_per_mile': 5.028858144819045,
  'category': 'normal'},
 '4197 - 41842 Royal Star': {'avg_trip_total': 16.05232180341334,
  'avg_trip_seconds': 862.5499680569501,
  'avg_total_per_second': 0.020349904675192158,
  'avg_total_per_mile': 6.303394898177784,
  'category': 'normal'},
 '5062 - 34841 Sam Mestas': {'avg_trip_total': 11.056639547223204,
  'avg_trip_seconds': 614.4747081712062,
  'avg_total_per_second': 0.02153168093565625,
  'avg_total_per_mile': 5.897313550055914,
  'category': 'normal'},
 "3591- Chuk's Cab": {'avg_trip_total': 11.187864864864865,
  'avg_trip_seconds': 741.8108108108108,
  'avg_total_per_second': 0.016458763700796835,
  'avg_total_per_mile': 6.680201188159931,
  'category': 'normal'},
 '2767 - Sayed M Badri': {'avg_trip_total': 10.87095238095238,
  'avg_trip_seconds': 515.4285714285713,
  'avg_total_per_second': 0.02476838672355756,
  'avg_total_per_mile': 47.480159116886725,
  'category': 'luxury'},
 '24 Seven Taxi': {'avg_trip_total': 18.030171207688856,
  'avg_trip_seconds': 935.8361582233331,
  'avg_total_per_second': 0.021259462791286786,
  'avg_total_per_mile': 6.850081154625212,
  'category': 'normal'},
 '3201 - CID Cab Co Inc': {'avg_trip_total': 12.976327800829877,
  'avg_trip_seconds': 785.3526970954357,
  'avg_total_per_second': 0.01872220137237825,
  'avg_total_per_mile': 5.615213774915291,
  'category': 'normal'},
 '3897 - Ilie Malec': {'avg_trip_total': 19.054652293384063,
  'avg_trip_seconds': 1038.50348763475,
  'avg_total_per_second': 0.019542657292566292,
  'avg_total_per_mile': 5.258555329805378,
  'category': 'normal'},
 '3623-Arrington Enterprises': {'avg_trip_total': 14.847528735632181,
  'avg_trip_seconds': 747.4005305039789,
  'avg_total_per_second': 0.020769422451026923,
  'avg_total_per_mile': 5.532585397162571,
  'category': 'normal'},
 '3152 - 97284 Crystal Abernathy': {'avg_trip_total': 10.959691075514872,
  'avg_trip_seconds': 700.1544622425627,
  'avg_total_per_second': 0.018329651838375403,
  'avg_total_per_mile': 4.399124549026014,
  'category': 'normal'},
 'Leonard Cab Co': {'avg_trip_total': 14.265727036779138,
  'avg_trip_seconds': 804.1071089447365,
  'avg_total_per_second': 0.02122618197537691,
  'avg_total_per_mile': 6.744522438564019,
  'category': 'normal'},
 'Yellow Cab': {'avg_trip_total': 14.7474711859302,
  'avg_trip_seconds': 804.9368234709988,
  'avg_total_per_second': 0.032016060035425176,
  'avg_total_per_mile': 6.057688892307081,
  'category': 'normal'},
 '6743 - Luhak Corp': {'avg_trip_total': 13.592444749920697,
  'avg_trip_seconds': 819.1582954425293,
  'avg_total_per_second': 0.017806451873724236,
  'avg_total_per_mile': 7.178549150997802,
  'category': 'normal'},
 '3201 - C&D Cab Co Inc': {'avg_trip_total': 12.506872584678339,
  'avg_trip_seconds': 744.8817913162081,
  'avg_total_per_second': 0.019236701949423625,
  'avg_total_per_mile': 6.298893415436767,
  'category': 'normal'},
 '5776 - Mekonen Cab Company': {'avg_trip_total': 13.225312631137216,
  'avg_trip_seconds': 778.8292068820816,
  'avg_total_per_second': 0.01860023950399831,
  'avg_total_per_mile': 5.757945032939189,
  'category': 'normal'},
 '3141 - 87803 Zip Cab': {'avg_trip_total': 12.60624589674724,
  'avg_trip_seconds': 682.9185317815577,
  'avg_total_per_second': 0.021194681985293846,
  'avg_total_per_mile': 6.498011673857313,
  'category': 'normal'},
 'Setare Inc': {'avg_trip_total': 15.504033230648231,
  'avg_trip_seconds': 826.8984728321421,
  'avg_total_per_second': 0.022334513734946198,
  'avg_total_per_mile': 6.974296076687397,
  'category': 'normal'},
 'Checker Taxi': {'avg_trip_total': 14.498088731376537,
  'avg_trip_seconds': 796.6284621864237,
  'avg_total_per_second': 0.05401611030583098,
  'avg_total_per_mile': 6.053222183661102,
  'category': 'normal'},
 '5006 - 39261 Salifu Bawa': {'avg_trip_total': 12.00953795379538,
  'avg_trip_seconds': 764.6720297029702,
  'avg_total_per_second': 0.01772520810680647,
  'avg_total_per_mile': 6.762485100955214,
  'category': 'normal'},
 '3623 - 72222 Arrington Enterprises': {'avg_trip_total': 18.15612516823688,
  'avg_trip_seconds': 848.842530282638,
  'avg_total_per_second': 0.021822062705768412,
  'avg_total_per_mile': 5.617186606154316,
  'category': 'normal'},
 'Norshore Cab': {'avg_trip_total': 10.702041264016287,
  'avg_trip_seconds': 798.3840244648324,
  'avg_total_per_second': 0.0203749950195822,
  'avg_total_per_mile': 4.611227756178642,
  'category': 'normal'},
 'Dispatch Taxi Affiliation': {'avg_trip_total': 14.237034545119592,
  'avg_trip_seconds': 777.2931672035853,
  'avg_total_per_second': 0.020294207107451365,
  'avg_total_per_mile': 8.034894603384931,
  'category': 'normal'},
 '5864 - 73614 Thomas Owusu': {'avg_trip_total': 13.22187755102041,
  'avg_trip_seconds': 829.3994169096209,
  'avg_total_per_second': 0.01818905478057998,
  'avg_total_per_mile': 5.511007371328713,
  'category': 'normal'},
 '3669 - 85800 Jordan Taxi Inc': {'avg_trip_total': 37.64937360178971,
  'avg_trip_seconds': 1788.8590604026847,
  'avg_total_per_second': 0.023830676513661393,
  'avg_total_per_mile': 7.767661731893406,
  'category': 'normal'},
 'Patriot Trans Inc': {'avg_trip_total': 23.23915088177662,
  'avg_trip_seconds': 1055.0359242325278,
  'avg_total_per_second': 0.025858599675026045,
  'avg_total_per_mile': 6.994922497424224,
  'category': 'normal'},
 'Northwest Management LLC': {'avg_trip_total': 13.733567853264038,
  'avg_trip_seconds': 780.7078737738475,
  'avg_total_per_second': 0.019635761377612245,
  'avg_total_per_mile': 8.116729218081128,
  'category': 'normal'},
 '6488 - 83287 Zuha Taxi': {'avg_trip_total': 11.406600947381898,
  'avg_trip_seconds': 712.902317244911,
  'avg_total_per_second': 0.018421566415820708,
  'avg_total_per_mile': 5.8525181649250255,
  'category': 'normal'},
 '0694 - 59280 Chinesco Trans Inc': {'avg_trip_total': 13.346400466645457,
  'avg_trip_seconds': 859.0836780146357,
  'avg_total_per_second': 0.01720455081991834,
  'avg_total_per_mile': 6.354678830842873,
  'category': 'normal'},
 '4732 - Maude Lamy': {'avg_trip_total': 29.0,
  'avg_trip_seconds': 845.0,
  'avg_total_per_second': 0.04162067294324354,
  'avg_total_per_mile': 16.2885811012357,
  'category': 'luxury'},
 '0694 - Chinesco Trans Inc': {'avg_trip_total': 15.185361867704284,
  'avg_trip_seconds': 916.6225680933852,
  'avg_total_per_second': 0.01787923728829166,
  'avg_total_per_mile': 6.106185838501131,
  'category': 'normal'},
 '5074 - Ahzmi Inc': {'avg_trip_total': 12.886179417122039,
  'avg_trip_seconds': 816.2295081967212,
  'avg_total_per_second': 0.018071733633359382,
  'avg_total_per_mile': 5.286561648720513,
  'category': 'normal'},
 'American United Taxi Affiliation': {'avg_trip_total': 23.36765893792072,
  'avg_trip_seconds': 1196.58937920718,
  'avg_total_per_second': 0.021142152669947886,
  'avg_total_per_mile': 7.008383132750617,
  'category': 'normal'},
 '4787 - 56058 Reny Cab Co': {'avg_trip_total': 11.31795036764706,
  'avg_trip_seconds': 695.1286764705883,
  'avg_total_per_second': 0.018605134322885655,
  'avg_total_per_mile': 5.548078795885162,
  'category': 'normal'},
 '0118 - 42111 Godfrey S.Awir': {'avg_trip_total': 16.311374715886814,
  'avg_trip_seconds': 903.0613684458029,
  'avg_total_per_second': 0.019257709368528605,
  'avg_total_per_mile': 6.2868118336826555,
  'category': 'normal'},
 '6488 - Zuha Taxi': {'avg_trip_total': 11.351755085759875,
  'avg_trip_seconds': 703.9648982848024,
  'avg_total_per_second': 0.018613631070442473,
  'avg_total_per_mile': 5.6588306114492735,
  'category': 'normal'},
 '4053 - 40193 Adwar H. Nikola': {'avg_trip_total': 25.9902825203252,
  'avg_trip_seconds': 1201.5609756097563,
  'avg_total_per_second': 0.02287515000918057,
  'avg_total_per_mile': 5.445622201644518,
  'category': 'normal'},
 '5724 - 75306 KYVI Cab Inc': {'avg_trip_total': 12.708960882867128,
  'avg_trip_seconds': 747.5415209790209,
  'avg_total_per_second': 0.01894570299494753,
  'avg_total_per_mile': 6.6965468280065465,
  'category': 'normal'},
 '6574 - Babylon Express Inc.': {'avg_trip_total': 15.033412425054216,
  'avg_trip_seconds': 847.7076157673174,
  'avg_total_per_second': 0.01960157177066992,
  'avg_total_per_mile': 6.624468444650083,
  'category': 'normal'},
 'Top Cab Affiliation': {'avg_trip_total': 15.032251338034351,
  'avg_trip_seconds': 822.5446434756101,
  'avg_total_per_second': 0.019888639820940912,
  'avg_total_per_mile': 6.0562031743864155,
  'category': 'normal'},
 'KOAM Taxi Association': {'avg_trip_total': 14.986722939416865,
  'avg_trip_seconds': 830.8409867057621,
  'avg_total_per_second': 0.019544049787259498,
  'avg_total_per_mile': 6.00575901060286,
  'category': 'normal'},
 '3011 - 66308 JBL Cab Inc.': {'avg_trip_total': 13.323767648864333,
  'avg_trip_seconds': 700.9898710865561,
  'avg_total_per_second': 0.021334249587941476,
  'avg_total_per_mile': 6.516568133516236,
  'category': 'normal'},
 '2192 - 73487 Zeymane Corp': {'avg_trip_total': 13.62646407449352,
  'avg_trip_seconds': 748.7353718384296,
  'avg_total_per_second': 0.020280338331052212,
  'avg_total_per_mile': 6.471766571284175,
  'category': 'normal'},
 '3011 - JBL Cab Inc.': {'avg_trip_total': 11.665342857142855,
  'avg_trip_seconds': 660.0428571428571,
  'avg_total_per_second': 0.01998940472224747,
  'avg_total_per_mile': 6.197199817239657,
  'category': 'normal'},
 'Chicago Taxicab': {'avg_trip_total': 18.915943495198327,
  'avg_trip_seconds': 987.4508719288083,
  'avg_total_per_second': 0.02151272832201967,
  'avg_total_per_mile': 7.237253485313515,
  'category': 'normal'},
 '4787 - Reny Cab Co': {'avg_trip_total': 10.557608558842038,
  'avg_trip_seconds': 642.1648835745755,
  'avg_total_per_second': 0.018834302439473517,
  'avg_total_per_mile': 10.656939136040423,
  'category': 'expensive'},
 '1085 - 72312 N and W Cab Co': {'avg_trip_total': 10.30251207320173,
  'avg_trip_seconds': 594.3065322375669,
  'avg_total_per_second': 0.019864855809214238,
  'avg_total_per_mile': 6.310825933925061,
  'category': 'normal'},
 '6057 - 24657 Richard Addo': {'avg_trip_total': 13.962673484295104,
  'avg_trip_seconds': 920.5222790357928,
  'avg_total_per_second': 0.016188282354776738,
  'avg_total_per_mile': 5.922264832727,
  'category': 'normal'},
 '0118 - Godfrey S.Awir': {'avg_trip_total': 13.675078534031414,
  'avg_trip_seconds': 856.9044502617803,
  'avg_total_per_second': 0.017233984062389945,
  'avg_total_per_mile': 6.117753297680383,
  'category': 'normal'},
 '5074 - 54002 Ahzmi Inc': {'avg_trip_total': 16.247116795974,
  'avg_trip_seconds': 942.1849444327946,
  'avg_total_per_second': 0.0185989881061378,
  'avg_total_per_mile': 5.544717127893473,
  'category': 'normal'},
 '3094 - 24059 G.L.B. Cab Co': {'avg_trip_total': 10.125193187595324,
  'avg_trip_seconds': 700.988815455008,
  'avg_total_per_second': 0.015835608647613412,
  'avg_total_per_mile': 5.772948067450157,
  'category': 'normal'},
 '5437 - Great American Cab Co': {'avg_trip_total': 12.370236946028959,
  'avg_trip_seconds': 735.1908731899955,
  'avg_total_per_second': 0.019119789840354947,
  'avg_total_per_mile': 6.672697613238649,
  'category': 'normal'},
 'Gold Coast Taxi': {'avg_trip_total': 18.27227599553806,
  'avg_trip_seconds': 970.5992984531025,
  'avg_total_per_second': 0.020642440934653304,
  'avg_total_per_mile': 7.094917763262297,
  'category': 'normal'},
 '3201 - C & D Cab Co Inc': {'avg_trip_total': 11.862983606557378,
  'avg_trip_seconds': 699.1475409836065,
  'avg_total_per_second': 0.01983792624175732,
  'avg_total_per_mile': 5.931469285845865,
  'category': 'normal'},
 '3591 - 63480 Chuks Cab': {'avg_trip_total': 11.553643844634628,
  'avg_trip_seconds': 714.0947992100064,
  'avg_total_per_second': 0.018165275868488855,
  'avg_total_per_mile': 7.055728868335003,
  'category': 'normal'},
 'American United': {'avg_trip_total': 14.446361813354697,
  'avg_trip_seconds': 811.8937034255063,
  'avg_total_per_second': 0.02928751589957487,
  'avg_total_per_mile': 5.954099926279735,
  'category': 'normal'},
 '3141 - Zip Cab': {'avg_trip_total': 14.100231389660186,
  'avg_trip_seconds': 777.6913923046411,
  'avg_total_per_second': 0.020351222312071603,
  'avg_total_per_mile': 5.39509474052506,
  'category': 'normal'},
 '3253 - 91138 Gaither Cab Co.': {'avg_trip_total': 12.06503134947429,
  'avg_trip_seconds': 697.6714575094048,
  'avg_total_per_second': 0.019849096952569242,
  'avg_total_per_mile': 6.723683050904951,
  'category': 'normal'},
 'Flash Cab': {'avg_trip_total': 14.997488243791857,
  'avg_trip_seconds': 813.3149082687731,
  'avg_total_per_second': 0.0258767438276246,
  'avg_total_per_mile': 5.9479958918173885,
  'category': 'normal'},
 '5997 - 65283 AW Services Inc.': {'avg_trip_total': 19.649642121041794,
  'avg_trip_seconds': 1063.4513771781901,
  'avg_total_per_second': 0.01988139933686589,
  'avg_total_per_mile': 6.183551347782214,
  'category': 'normal'},
 '1247 - Daniel Ayertey': {'avg_trip_total': 15.222663129973476,
  'avg_trip_seconds': 920.1485411140582,
  'avg_total_per_second': 0.01783424155305932,
  'avg_total_per_mile': 5.0042377996282275,
  'category': 'normal'},
 '5724 - KYVI Cab Inc': {'avg_trip_total': 11.811361440491876,
  'avg_trip_seconds': 748.9855072463768,
  'avg_total_per_second': 0.017554325663240476,
  'avg_total_per_mile': 5.394637030056906,
  'category': 'normal'},
 '3253 - Gaither Cab Co.': {'avg_trip_total': 13.610890688259106,
  'avg_trip_seconds': 803.4412955465586,
  'avg_total_per_second': 0.019044553205423784,
  'avg_total_per_mile': 6.126627567716028,
  'category': 'normal'},
 '6747 - Mueen Abdalla': {'avg_trip_total': 18.09917142204976,
  'avg_trip_seconds': 928.2264323213877,
  'avg_total_per_second': 0.020935803595556794,
  'avg_total_per_mile': 6.382491445980868,
  'category': 'normal'},
 '2241 - 44667 - Felman Corp, Manuel Alonso': {'avg_trip_total': 40.75369458128079,
  'avg_trip_seconds': 1943.9408866995072,
  'avg_total_per_second': 0.02266025239257348,
  'avg_total_per_mile': 4.237514268683283,
  'category': 'normal'},
 'Chicago Medallion Leasing INC': {'avg_trip_total': 14.831340209587653,
  'avg_trip_seconds': 787.6106876106876,
  'avg_total_per_second': 0.020599165799549242,
  'avg_total_per_mile': 7.898192562657727,
  'category': 'normal'},
 "3591 - 63480 Chuk's Cab": {'avg_trip_total': 10.275000000000002,
  'avg_trip_seconds': 647.0588235294118,
  'avg_total_per_second': 0.01776516356075077,
  'avg_total_per_mile': 6.912482785436541,
  'category': 'normal'},
 'Medallion Leasin': {'avg_trip_total': 17.20772892486458,
  'avg_trip_seconds': 883.0048884382498,
  'avg_total_per_second': 0.02139429931122795,
  'avg_total_per_mile': 7.164310219092372,
  'category': 'normal'},
 'Chicago Medallion Management': {'avg_trip_total': 12.544761698325205,
  'avg_trip_seconds': 711.763264295004,
  'avg_total_per_second': 0.019637109026629504,
  'avg_total_per_mile': 6.447150508416058,
  'category': 'normal'},
 '1085 - N and W Cab Co': {'avg_trip_total': 9.921085487077539,
  'avg_trip_seconds': 588.5884691848908,
  'avg_total_per_second': 0.019204023474096627,
  'avg_total_per_mile': 5.830721905480584,
  'category': 'normal'},
 '4053 - Adwar H. Nikola': {'avg_trip_total': 16.77387140902873,
  'avg_trip_seconds': 933.1805745554035,
  'avg_total_per_second': 0.019653590671137585,
  'avg_total_per_mile': 6.096672914137409,
  'category': 'normal'},
 '2092 - 61288 Sbeih company': {'avg_trip_total': 14.523229771252993,
  'avg_trip_seconds': 882.5059747354046,
  'avg_total_per_second': 0.017454846924167013,
  'avg_total_per_mile': 5.777726421076896,
  'category': 'normal'},
 '3385 - 23210  Eman Cab': {'avg_trip_total': 16.30723577235773,
  'avg_trip_seconds': 1020.5853658536586,
  'avg_total_per_second': 0.01797112395015759,
  'avg_total_per_mile': 5.797859124964721,
  'category': 'normal'},
 'Chicago Independents': {'avg_trip_total': 18.842194537972304,
  'avg_trip_seconds': 986.7129068462401,
  'avg_total_per_second': 0.021268738164004083,
  'avg_total_per_mile': 8.765010123372432,
  'category': 'normal'},
 '3669 - Jordan Taxi Inc': {'avg_trip_total': 24.233333333333334,
  'avg_trip_seconds': 1500.0,
  'avg_total_per_second': 0.03202169153518479,
  'avg_total_per_mile': 80.69530423280423,
  'category': 'luxury'},
 'Taxi Affiliation Service Yellow': {'avg_trip_total': 19.892497398236646,
  'avg_trip_seconds': 887.8701338863892,
  'avg_total_per_second': 0.027812581339117858,
  'avg_total_per_mile': 9.189868232191406,
  'category': 'expensive'},
 'NULL': {'avg_trip_total': 14.629240406168082,
  'avg_trip_seconds': 823.4628996487664,
  'avg_total_per_second': 0.020955887440934254,
  'avg_total_per_mile': 12.7938563355302,
  'category': 'expensive'},
 '2823 - 73307 Seung Lee': {'avg_trip_total': 23.964766606822263,
  'avg_trip_seconds': 1350.2558348294435,
  'avg_total_per_second': 0.018292485856572414,
  'avg_total_per_mile': 5.357169553962938,
  'category': 'normal'},
 '6743 - 78771 Luhak Corp': {'avg_trip_total': 13.454379036448673,
  'avg_trip_seconds': 858.4886051518077,
  'avg_total_per_second': 0.016757166031661882,
  'avg_total_per_mile': 5.934023873363925,
  'category': 'normal'},
 '3385 - Eman Cab': {'avg_trip_total': 15.091637010676157,
  'avg_trip_seconds': 928.2918149466191,
  'avg_total_per_second': 0.018560130598127604,
  'avg_total_per_mile': 5.404043192904852,
  'category': 'normal'},
 '3556 - 36214 RC Andrews Cab': {'avg_trip_total': 15.291086885779375,
  'avg_trip_seconds': 996.1601041327691,
  'avg_total_per_second': 0.016478904622824784,
  'avg_total_per_mile': 4.926863144019903,
  'category': 'normal'},
 '4623 - Jay Kim': {'avg_trip_total': 16.358426395939087,
  'avg_trip_seconds': 974.7715736040609,
  'avg_total_per_second': 0.01890407313452918,
  'avg_total_per_mile': 5.633315633033479,
  'category': 'normal'},
 'Nova Taxi Affiliation Llc': {'avg_trip_total': 14.073822874944382,
  'avg_trip_seconds': 823.0147134955455,
  'avg_total_per_second': 0.04309119246847618,
  'avg_total_per_mile': 8.549348498340807,
  'category': 'normal'},
 'Suburban Dispatch LLC': {'avg_trip_total': 26.266666666666666,
  'avg_trip_seconds': 220.0,
  'avg_total_per_second': 0.31747222222222216,
  'avg_total_per_mile': 18.561666666666667,
  'category': 'luxury'},
 '3385 - 23210 Eman Cab': {'avg_trip_total': 14.493386243386244,
  'avg_trip_seconds': 852.8042328042329,
  'avg_total_per_second': 0.018331313285263724,
  'avg_total_per_mile': 5.420053590368147,
  'category': 'normal'},
 'City Service': {'avg_trip_total': 17.21579288263372,
  'avg_trip_seconds': 879.9324964467238,
  'avg_total_per_second': 0.026129935447225916,
  'avg_total_per_mile': 7.114092702971301,
  'category': 'normal'},
 'Service Taxi Association': {'avg_trip_total': 18.096512126823534,
  'avg_trip_seconds': 948.0262860756065,
  'avg_total_per_second': 0.02109453222994594,
  'avg_total_per_mile': 6.743273592026298,
  'category': 'normal'},
 "3591- 63480 Chuk's Cab": {'avg_trip_total': 10.782501013376569,
  'avg_trip_seconds': 690.6931495743817,
  'avg_total_per_second': 0.017338542896099,
  'avg_total_per_mile': 6.6998303444403255,
  'category': 'normal'},
 '3620 - 52292 David K. Cab Corp.': {'avg_trip_total': 20.51022311942202,
  'avg_trip_seconds': 1113.557161070973,
  'avg_total_per_second': 0.019182751393540982,
  'avg_total_per_mile': 6.073833526409391,
  'category': 'normal'},
 '3094 - G.L.B. Cab Co': {'avg_trip_total': 11.075518134715027,
  'avg_trip_seconds': 772.9274611398963,
  'avg_total_per_second': 0.01582780624532871,
  'avg_total_per_mile': 4.906927248009473,
  'category': 'normal'},
 '2809 - 95474 C & D Cab Co Inc.': {'avg_trip_total': 16.44862798172899,
  'avg_trip_seconds': 793.3581458298086,
  'avg_total_per_second': 0.022051590616123534,
  'avg_total_per_mile': 6.490363085990298,
  'category': 'normal'},
 'Chicago Carriage Cab Corp': {'avg_trip_total': 17.091944068580492,
  'avg_trip_seconds': 927.1394563896276,
  'avg_total_per_second': 0.0819900877906878,
  'avg_total_per_mile': 6.905701364764062,
  'category': 'normal'},
 '3319 - CD Cab Co': {'avg_trip_total': 15.818772869254795,
  'avg_trip_seconds': 804.7121820615796,
  'avg_total_per_second': 0.021948809732954246,
  'avg_total_per_mile': 6.980511300086034,
  'category': 'normal'},
 '303 Taxi Waukegan': {'avg_trip_total': 9.866586248492158,
  'avg_trip_seconds': 652.6682750301569,
  'avg_total_per_second': 0.029056726342346913,
  'avg_total_per_mile': 3.8211819006607803,
  'category': 'normal'},
 'Choice Taxi Association': {'avg_trip_total': 15.464780801693408,
  'avg_trip_seconds': 839.9161639470236,
  'avg_total_per_second': 0.02003893426645762,
  'avg_total_per_mile': 8.100906104282988,
  'category': 'normal'},
 'C & D Cab Co Inc': {'avg_trip_total': 13.061418289991366,
  'avg_trip_seconds': 770.8741963343249,
  'avg_total_per_second': 0.01852590585485208,
  'avg_total_per_mile': 5.518807280925552,
  'category': 'normal'},
 '3897 - 57856 Ilie Malec': {'avg_trip_total': 17.241948424068767,
  'avg_trip_seconds': 994.9641833810888,
  'avg_total_per_second': 0.018618590490296804,
  'avg_total_per_mile': 5.732977388308638,
  'category': 'normal'},
 '2192 - Zeymane Corp': {'avg_trip_total': 11.713334064007014,
  'avg_trip_seconds': 675.1380973257345,
  'avg_total_per_second': 0.019780053346139227,
  'avg_total_per_mile': 18.693121220173165,
  'category': 'expensive'},
 '5006 - Salifu Bawa': {'avg_trip_total': 11.170304054054053,
  'avg_trip_seconds': 761.7567567567568,
  'avg_total_per_second': 0.01654281682102606,
  'avg_total_per_mile': 6.303825278576794,
  'category': 'normal'},
 '4197 - Royal Star': {'avg_trip_total': 14.082722948870392,
  'avg_trip_seconds': 881.1652794292509,
  'avg_total_per_second': 0.018220893343692906,
  'avg_total_per_mile': 5.53024264155648,
  'category': 'normal'},
 '5062 - Sam Mestas': {'avg_trip_total': 10.44247448979592,
  'avg_trip_seconds': 652.1938775510204,
  'avg_total_per_second': 0.019415914372791304,
  'avg_total_per_mile': 5.742362216878544,
  'category': 'normal'},
 '4615 - 83503 Tyrone Henderson': {'avg_trip_total': 12.647129467831611,
  'avg_trip_seconds': 792.5718824463863,
  'avg_total_per_second': 0.018772833407691744,
  'avg_total_per_mile': 6.196008337859875,
  'category': 'normal'},
 '4623 - 27290 Jay Kim': {'avg_trip_total': 20.640474804919982,
  'avg_trip_seconds': 1033.966406559979,
  'avg_total_per_second': 0.02143939672425555,
  'avg_total_per_mile': 5.987742044569887,
  'category': 'normal'},
 '5129 - Mengisti Taxi': {'avg_trip_total': 12.318688193743695,
  'avg_trip_seconds': 782.4823410696265,
  'avg_total_per_second': 0.017691660182391152,
  'avg_total_per_mile': 6.3254122744431625,
  'category': 'normal'},
 '5874 - 73628 Sergey Cab Corp.': {'avg_trip_total': 19.483402230078866,
  'avg_trip_seconds': 878.2975251563774,
  'avg_total_per_second': 0.022588952997588105,
  'avg_total_per_mile': 5.452721731202245,
  'category': 'normal'},
 '6742 - 83735 Tasha ride inc': {'avg_trip_total': 16.13564597564598,
  'avg_trip_seconds': 947.0804870804868,
  'avg_total_per_second': 0.018245533555718806,
  'avg_total_per_mile': 6.848373832731597,
  'category': 'normal'},
 '3385 -  Eman Cab': {'avg_trip_total': 14.723557692307693,
  'avg_trip_seconds': 969.2307692307692,
  'avg_total_per_second': 0.01777512590883782,
  'avg_total_per_mile': 5.807423007266183,
  'category': 'normal'},
 'Metro Jet Taxi A': {'avg_trip_total': 20.1454686899731,
  'avg_trip_seconds': 996.2754514022281,
  'avg_total_per_second': 0.02195652406322318,
  'avg_total_per_mile': 6.520898056276227,
  'category': 'normal'},
 '1247 - 72807 Daniel Ayertey': {'avg_trip_total': 15.133752069764874,
  'avg_trip_seconds': 898.5627552710013,
  'avg_total_per_second': 0.018273983847047398,
  'avg_total_per_mile': 5.303911201534807,
  'category': 'normal'},
 '2823 - 73307 Lee Express Inc': {'avg_trip_total': 32.33858746492049,
  'avg_trip_seconds': 1623.4798877455564,
  'avg_total_per_second': 0.02043847247138236,
  'avg_total_per_mile': 5.856743664553451,
  'category': 'normal'},
 '5997 - AW Services Inc.': {'avg_trip_total': 16.51214162348877,
  'avg_trip_seconds': 988.3592400690843,
  'avg_total_per_second': 0.01819939650861908,
  'avg_total_per_mile': 5.605967123321373,
  'category': 'normal'},
 '2241 - Manuel Alonso': {'avg_trip_total': 28.840433403805502,
  'avg_trip_seconds': 1644.1014799154336,
  'avg_total_per_second': 0.018503033045209078,
  'avg_total_per_mile': 8.385747781696024,
  'category': 'normal'},
 '2733 - Benny Jona': {'avg_trip_total': 15.949270270270272,
  'avg_trip_seconds': 928.3659043659047,
  'avg_total_per_second': 0.01809765087446865,
  'avg_total_per_mile': 7.750018857684225,
  'category': 'normal'},
 '3319 - C&D Cab Company': {'avg_trip_total': 12.254405381944444,
  'avg_trip_seconds': 745.8854166666666,
  'avg_total_per_second': 0.017413723532301056,
  'avg_total_per_mile': 5.422129542070558,
  'category': 'normal'}}

class TaxiTripTotalReduced2017FullV04Run(TaxiTripTotalReduced2017FullV02Run):
    flow_dir   = "TaxiTripTotalReduced2017FullV04"
    model_name = "20210518"
    query      = """SELECT 
        # IDs, these will not be features, we keep them for record traceability
        unique_key,
        #taxi_id,

        # raw features
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,
        FORMAT_TIMESTAMP("%Y-%m-%d %a-%H:%M", trip_start_timestamp, "UTC") as trip_start,
        company,

        # labels
        #trip_seconds,
        trip_total,

        # data split
        case EXTRACT(YEAR FROM trip_start_timestamp)
            when 2018 then "VALIDATE"
            when 2019 then "TEST"
            when 2020 then "COVID19"
            when 2021 then "COVID19"
            else "TRAIN"
        END as split_set

        # fields not availbale at prediction time: trip_end_timestamp, fare, tips, tolls, extras, payment_type

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        where
        # reproducibility constraints
            trip_start_timestamp <= TIMESTAMP("2021-03-01 00:00:00 UTC")
        AND trip_start_timestamp >= TIMESTAMP("2013-01-01 00:00:00 UTC")

        # label constraints
        AND trip_seconds is not null
        AND trip_seconds > 0
        AND trip_total   is not null
        AND trip_total   > 0

        # feature nullability constraints
        AND trip_miles             is not null
        AND pickup_census_tract    is not null
        AND dropoff_census_tract   is not null
        AND pickup_community_area  is not null
        AND dropoff_community_area is not null
        AND company                is not null
        AND pickup_latitude        is not null
        AND pickup_longitude       is not null
        AND dropoff_latitude       is not null
        AND dropoff_longitude      is not null
        AND EXTRACT(YEAR from trip_start_timestamp) >= 2017 # train set limited to year 2017
        and trip_miles   > 0 
        AND trip_seconds > 0 
        AND trip_total   > 0  
        AND ABS(dropoff_latitude-pickup_latitude) + ABS(dropoff_longitude-pickup_longitude) > 0 # L1 distance
    """
    key_column          = "unique_key"
    label_column        = "trip_total"
    split_column        = "split_set"
    numeric_columns     = ['pickup_latitude',
                           'pickup_longitude',
                           'dropoff_latitude',
                           'dropoff_longitude']
    categorical_columns = ['trip_start', # [technical] this is in categorical because its TF placeholder for serving must be of type string
                           'company']
    
    
    PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS = 20
    
    def preprocess_fn(self, input_features):
        NUM_BUCKETS = self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS
        
        output_features = {}
        output_features[self.label_column] = input_features[self.label_column]
        
        for c in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']:
            output_features[f'{c}_bucketized'] = tft.bucketize(input_features[c], NUM_BUCKETS)
            
        # date preprocessing
        def tf_strings_split(s, sep, len):
            return [tf.strings.split(s, sep=sep).to_tensor()[:,l] for l in range(len)]
        
        date_and_time             = input_features['trip_start']
        date, day_and_time        = tf_strings_split(date_and_time, sep=" ", len=2)
        year, month, day_of_month = tf_strings_split(date,          sep="-", len=3)
        day_of_week, time         = tf_strings_split(day_and_time,  sep="-", len=2)
        hour, minutes             = tf_strings_split(time,          sep=":", len=2)
        
        output_features[f'month_index']        = tft.compute_and_apply_vocabulary(month,        vocab_filename='month')
        output_features[f'day_of_month_index'] = tft.compute_and_apply_vocabulary(day_of_month, vocab_filename='day_of_month')
        output_features[f'day_of_week_index']  = tft.compute_and_apply_vocabulary(day_of_week,  vocab_filename='day_of_week')
        output_features[f'hour_index']         = tft.compute_and_apply_vocabulary(hour,         vocab_filename='hour')
            
        #distance preprocessing
        l1_distance =   tf.abs(input_features['pickup_latitude']-input_features['dropoff_latitude']
                      )+tf.abs(input_features['pickup_longitude']-input_features['dropoff_longitude'])
        output_features['l1_distance'] = tft.scale_to_z_score(l1_distance)
        
        companies = list(sorted(company_historical_data.keys()))
        
        #company avg_total_per_second
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(companies), 
                [company_historical_data[c]["avg_total_per_second"] for c in companies]),
            default_value=0)
        output_features['company_avg_total_per_second'] = table.lookup(input_features["company"])
        
        #company avg_total_per_mile
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(companies), 
                [company_historical_data[c]["avg_total_per_mile"] for c in companies]),
            default_value=0)
        output_features['company_avg_total_per_mile'] = table.lookup(input_features["company"])
        
        #company category
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(companies), 
                [company_historical_data[c]["category"] for c in companies]),
            default_value="")
        output_features['company_category'] = table.lookup(input_features["company"])
        
        return output_features