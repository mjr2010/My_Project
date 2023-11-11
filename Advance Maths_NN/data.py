import pandas as pd
import numpy as np
import random
import torch

proteins = pd.DataFrame(columns = ['amino acid','gene', 'protein'])

# Add Hemoglobin proteins in human body
proteins.loc[0] = ['MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK\
VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG\
KEFTPPVQAAYQKVVAGVANALAHKYH', 'HBB', 'Hemoglobin subunit beta']
proteins.loc[1] = ['MVHLTPEEKTAVNALWGKVNVDAVGGEALGRLLVVYPWTQRFFESFGDLSSPDAVMGNPK\
VKAHGKKVLGAFSDGLAHLDNLKGTFSQLSELHCDKLHVDPENFRLLGNVLVCVLARNFG\
KEFTPQMQAAYQKVVAGVANALAHKYH', 'HBD', 'Hemoglobin subunit delta']
proteins.loc[2] = ['MVHFTAEEKAAVTSLWSKMNVEEAGGEALGRLLVVYPWTQRFFDSFGNLSSPSAILGNPK\
VKAHGKKVLTSFGDAIKNMDNLKPAFAKLSELHCDKLHVDPENFKLLGNVMVIILATHFG\
KEFTPEVQAAWQKLVSAVAIALAHKYH', 'HBE1', 'Hemoglobin subunit epsilon']
proteins.loc[3] = ['MSLTKTERTIIVSMWAKISTQADTIGTETLERLFLSHPQTKTYFPHFDLHPGSAQLRAHG\
SKVVAAVGDAVKSIDDIGGALSKLSELHAYILRVDPVNFKLLSHCLLVTLAARFPADFTA\
EAHAAWDKFLSVVSSVLTEKYR', 'HBZ', 'Hemoglobin subunit zeta']
proteins.loc[4] = ['MGHFTEEDKATITSLWGKVNVEDAGGETLGRLLVVYPWTQRFFDSFGNLSSASAIMGNPK\
VKAHGKKVLTSLGDAIKHLDDLKGTFAQLSELHCDKLHVDPENFKLLGNVLVTVLAIHFG\
KEFTPEVQASWQKMVTGVASALSSRYH', 'HBG2', 'Hemoglobin subunit gamma-2']
proteins.loc[5] = ['MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG\
KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP\
AVHASLDKFLASVSTVLTSKYR', 'HBA1', 'Hemoglobin subunit alpha']

# Add Insulin proteins in human body
proteins.loc[6] = ['MATGGRRGAAAAPLLVAVAALLLGAAGHLYPGEVCPGMDIRNNLTRLHELENCSVIEGHL\
QILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYAL\
VIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNYIVLNKDDNE\
ECGDICPGTAKGKTNCPATVINGQFVERCWTHSHCQKVCPTICKSHGCTAEGLCCHSECL\
GNCSQPDDPTKCVACRNFYLDGRCVETCPPPYYHFQDWRCVNFSFCQDLHHKCKNSRRQG\
CHQYVIHNNKCIPECPSGYTMNSSNLLCTPCLGPCPKVCHLLEGEKTIDSVTSAQELRGC\
TVINGSLIINIRGGNNLAAELEANLGLIEEISGYLKIRRSYALVSLSFFRKLRLIRGETL\
EIGNYSFYALDNQNLRQLWDWSKHNLTITQGKLFFHYNPKLCLSEIHKMEEVSGTKGRQE\
RNDIALKTNGDQASCENELLKFSYIRTSFDKILLRWEPYWPPDFRDLLGFMLFYKEAPYQ\
NVTEFDGQDACGSNSWTVVDIDPPLRSNDPKSQNHPGWLMRGLKPWTQYAIFVKTLVTFS\
DERRTYGAKSDIIYVQTDATNPSVPLDPISVSNSSSQIILKWKPPSDPNGNITHYLVFWE\
RQAEDSELFELDYCLKGLKLPSRTWSPPFESEDSQKHNQSEYEDSAGECCSCPKTDSQIL\
KELEESSFRKTFEDYLHNVVFVPRKTSSGTGAEDPRPSRKRRSLGDVGNVTVAVPTVAAF\
PNTSSTSVPTSPEEHRPFEKVVNKESLVISGLRHFTGYRIELQACNQDTPEERCSVAAYV\
SARTMPEAKADDIVGPVTHEIFENNVVHLMWQEPKEPNGLIVLYEVSYRRYGDEELHLCV\
SRKHFALERGCRLRGLSPGNYSVRIRATSLAGNGSWTEPTYFYVTDYLDVPSNIAKIIIG\
PLIFVFLFSVVIGSIYLFLRKRQPDGPLGPLYASSNPEYLSASDVFPCSVYVPDEWEVSR\
EKITLLRELGQGSFGMVYEGNARDIIKGEAETRVAVKTVNESASLRERIEFLNEASVMKG\
FTCHHVVRLLGVVSKGQPTLVVMELMAHGDLKSYLRSLRPEAENNPGRPPPTLQEMIQMA\
AEIADGMAYLNAKKFVHRDLAARNCMVAHDFTVKIGDFGMTRDIYETDYYRKGGKGLLPV\
RWMAPESLKDGVFTTSSDMWSFGVVLWEITSLAEQPYQGLSNEQVLKFVMDGGYLDQPDN\
CPERVTDLMRMCWQFNPKMRPTFLEIVNLLKDDLHPSFPEVSFFHSEENKAPESEELEME\
FEDMENVPLDRSSHCQREEAGGRDGGSSLGFKRSYEEHIPYTHMNGGKKNGRILTLPRSN\
PS', 'INSR', 'Insulin receptor']
proteins.loc[6] = ['MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAIKRIGNHITKSPED\
KREYRGLELANGIKVLLISDPTTDKSSAALDVHIGSLSDPPNIAGLSHFCEHMLFLGTKK\
YPKENEYSQFLSEHAGSSNAFTSGEHTNYYFDVSHEHLEGALDRFAQFFLCPLFDESCKD\
REVNAVDSEHEKNVMNDAWRLFQLEKATGNPKHPFSKFGTGNKYTLETRPNQEGIDVRQE\
LLKFHSAYYSSNLMAVCVLGRESLDDLTNLVVKLFSEVENKNVPLPEFPEHPFQEEHLKQ\
LYKIVPIKDIRNLYVTFPIPDLQKYYKSNPGHYLGHLIGHEGPGSLLSELKSKGWVNTLV\
GGQKEGARGFMFFIINVDLTEEGLLHVEDIILHMFQYIQKLRAEGPQEWVFQECKDLNAV\
AFRFKDKERPRGYTSKIAGILHYYPLEEVLTAEYLLEEFRPDLIEMVLDKLRPENVRVAI\
VSKSFEGKTDRTEEWYGTQYKQEAIPDEVIKKWQNADLNGKFKLPTKNEFIPTNFEILPL\
EKEATPYPALIKDTAMSKLWFKQDDKFFLPKACLNFEFFSPFAYVDPLHCNMAYLYLELL\
KDSLNEYAYAAELAGLSYDLQNTIYGMYLSVKGYNDKQPILLKKIIEKMATFEIDEKRFE\
IIKEAYMRSLNNFRAEQPHQHAMYYLRLLMTEVAWTKDELKEALDDVTLPRLKAFIPQLL\
SRLHIEALLHGNITKQAALGIMQMVEDTLIEHAHTKPLLPSQLVRYREVQLPDRGWFVYQ\
QRNEVHNNCGIEIYYQTDMQSTSENMFLELFCQIISEPCFNTLRTKEQLGYIVFSGPRRA\
NGIQGLRFIIQSEKPPHYLESRVEAFLITMEKSIEDMTEEAFQKHIQALAIRRLDKPKKL\
SAECAKYWGEIISQQYNFDRDNTEVAYLKTLTKEDIIKFYKEMLAVDAPRRHKVSVHVLA\
REMDSCPVVGEFPCQNDINLSQAPALPQPEVIQNMTEFKRGLPLFPLVKPHINFMAAKL', 'IDE', 'Insulin-degrading enzyme']
proteins.loc[7] = ['MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED\
LQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN', 'INS', 'Insulin']

# Add Actin proteins in human body
proteins.loc[7] = ['MAPHRPAPALLCALSLALCALSLPVRAATASRGASQAGAPQGRVPEARPNSMVVEHPEFL\
KAGKEPGLQIWRVEKFDLVPVPTNLYGDFFTGDAYVILKTVQLRNGNLQYDLHYWLGNEC\
SQDESGAAAIFTVQLDDYLNGRAVQHREVQGFESATFLGYFKSGLKYKKGGVASGFKHVV\
PNEVVVQRLFQVKGRRVVRATEVPVSWESFNNGDCFILDLGNNIHQWCGSNSNRYERLKA\
TQVSKGIRDNERSGRARVHVSEEGTEPEAMLQVLGPKPALPAGTEDTAKEDAANRKLAKL\
YKVSNGAGTMSVSLVADENPFAQGALKSEDCFILDHGKDGKIFVWKGKQANTEERKAALK\
TASDFITKMDYPKQTQVSVLPEGGETPLFKQFFKNWRDPDQTDGLGLSYLSSHIANVERV\
PFDAATLHTSTAMAAQHGMDDDGTGQKQIWRIEGSNKVPVDPATYGQFYGGDSYIILYNY\
RHGGRQGQIIYNWQGAQSTQDEVAASAILTAQLDEELGGTPVQSRVVQGKEPAHLMSLFG\
GKPMIIYKGGTSREGGQTAPASTRLFQVRANSAGATRAVEVLPKAGALNSNDAFVLKTPS\
AAYLWVGTGASEAEKTGAQELLRVLRAQPVQVAEGSEPDGFWEALGGKAAYRTSPRLKDK\
KMDAHPPRLFACSNKIGRFVIEEVPGELMQEDLATDDVMLLDTWDQVFVWVGKDSQEEEK\
TEALTSAKRYIETDPANRDRRTPITVVKQGFEPPSFVGWFLGWDDDYWSVDPLDRAMAEL\
AA', 'GSN', 'Gelsolin']
proteins.loc[8] = ['MEEVPGDALCEHFEANILTQNRCQNCFHPEEAHGARYQELRSPSGAEVPYCDLPRCPPAP\
EDPLSASTSGCQSVVDPGLRPGPKRGPSPSAGLPEEGPTAAPRSRSRELEAVPYLEGLTT\
SLCGSCNEDPGSDPTSSPDSATPDDTSNSSSVDWDTVERQEEEAPSWDELAVMIPRRPRE\
GPRADSSQRAPSLLTRSPVGGDAAGQKKEDTGGGGRSAGQHWARLRGESGLSLERHRSTL\
TQASSMTPHSGPRSTTSQASPAQRDTAQAASTREIPRASSPHRITQRDTSRASSTQQEIS\
RASSTQQETSRASSTQEDTPRASSTQEDTPRASSTQWNTPRASSPSRSTQLDNPRTSSTQ\
QDNPQTSFPTCTPQRENPRTPCVQQDDPRASSPNRTTQRENSRTSCAQRDNPKASRTSSP\
NRATRDNPRTSCAQRDNPRASSPSRATRDNPTTSCAQRDNPRASRTSSPNRATRDNPRTS\
CAQRDNPRASSPSRATRDNPTTSCAQRDNPRASRTSSPNRATRDNPRTSCAQRDNPRASS\
PNRAARDNPTTSCAQRDNPRASRTSSPNRATRDNPRTSCAQRDNPRASSPNRATRDNPTT\
SCAQRDNPRASRTSSPNRATRDNPRTSCAQRDNPRASSPNRTTQQDSPRTSCARRDDPRA\
SSPNRTIQQENPRTSCALRDNPRASSPSRTIQQENPRTSCAQRDDPRASSPNRTTQQENP\
RTSCARRDNPRASSRNRTIQRDNPRTSCAQRDNPRASSPNRTIQQENLRTSCTRQDNPRT\
SSPNRATRDNPRTSCAQRDNLRASSPIRATQQDNPRTCIQQNIPRSSSTQQDNPKTSCTK\
RDNLRPTCTQRDRTQSFSFQRDNPGTSSSQCCTQKENLRPSSPHRSTQWNNPRNSSPHRT\
NKDIPWASFPLRPTQSDGPRTSSPSRSKQSEVPWASIALRPTQGDRPQTSSPSRPAQHDP\
PQSSFGPTQYNLPSRATSSSHNPGHQSTSRTSSPVYPAAYGAPLTSPEPSQPPCAVCIGH\
RDAPRASSPPRYLQHDPFPFFPEPRAPESEPPHHEPPYIPPAVCIGHRDAPRASSPPRHT\
QFDPFPFLPDTSDAEHQCQSPQHEPLQLPAPVCIGYRDAPRASSPPRQAPEPSLLFQDLP\
RASTESLVPSMDSLHECPHIPTPVCIGHRDAPSFSSPPRQAPEPSLFFQDPPGTSMESLA\
PSTDSLHGSPVLIPQVCIGHRDAPRASSPPRHPPSDLAFLAPSPSPGSSGGSRGSAPPGE\
TRHNLEREEYTVLADLPPPRRLAQRQPGPQAQCSSGGRTHSPGRAEVERLFGQERRKSEA\
AGAFQAQDEGRSQQPSQGQSQLLRRQSSPAPSRQVTMLPAKQAELTRRSQAEPPHPWSPE\
KRPEGDRQLQGSPLPPRTSARTPERELRTQRPLESGQAGPRQPLGVWQSQEEPPGSQGPH\
RHLERSWSSQEGGLGPGGWWGCGEPSLGAAKAPEGAWGGTSREYKESWGQPEAWEEKPTH\
ELPRELGKRSPLTSPPENWGGPAESSQSWHSGTPTAVGWGAEGACPYPRGSERRPELDWR\
DLLGLLRAPGEGVWARVPSLDWEGLLELLQARLPRKDPAGHRDDLARALGPELGPPGTND\
VPEQESHSQPEGWAEATPVNGHSPALQSQSPVQLPSPACTSTQWPKIKVTRGPATATLAG\
LEQTGPLGSRSTAKGPSLPELQFQPEEPEESEPSRGQDPLTDQKQADSADKRPAEGKAGS\
PLKGRLVTSWRMPGDRPTLFNPFLLSLGVLRWRRPDLLNFKKGWMSILDEPGEPPSPSLT\
TTSTSQWKKHWFVLTDSSLKYYRDSTAEEADELDGEIDLRSCTDVTEYAVQRNYGFQIHT\
KDAVYTLSAMTSGIRRNWIEALRKTVRPTSAPDVTKLSDSNKENALHSYSTQKGPLKAGE\
QRAGSEVISRGGPRKADGQRQALDYVELSPLTQASPQRARTPARTPDRLAKQEELERDLA\
QRSEERRKWFEATDSRTPEVPAGEGPRRGLGAPLTEDQQNRLSEEIEKKWQELEKLPLRE\
NKRVPLTALLNQSRGERRGPPSDGHEALEKEVQALRAQLEAWRLQGEAPQSALRSQEDGH\
IPPGYISQEACERSLAEMESSHQQVMEELQRHHERELQRLQQEKEWLLAEETAATASAIE\
AMKKAYQEELSRELSKTRSLQQGPDGLRKQHQSDVEALKRELQVLSEQYSQKCLEIGALM\
RQAEEREHTLRRCQQEGQELLRHNQELHGRLSEEIDQLRGFIASQGMGNGCGRSNERSSC\
ELEVLLRVKENELQYLKKEVQCLRDELQMMQKDKRFTSGKYQDVYVELSHIKTRSEREIE\
QLKEHLRLAMAALQEKESMRNSLAE', 'TRIOBP', 'TRIO and F-actin-binding protein']
proteins.loc[9] = ['MGKKSRVKTQKSGTGATATVSPKEILNLTSELLQKCSSPAPGPGKEWEEYVQIRTLVEKI\
RKKQKGLSVTFDGKREDYFPDLMKWASENGASVEGFEMVNFKEEGFGLRATRDIKAEELF\
LWVPRKLLMTVESAKNSVLGPLYSQDRILQAMGNIALAFHLLCERASPNSFWQPYIQTLP\
SEYDTPLYFEEDEVRYLQSTQAIHDVFSQYKNTARQYAYFYKVIQTHPHANKLPLKDSFT\
YEDYRWAVSSVMTRQNQIPTEDGSRVTLALIPLWDMCNHTNGLITTGYNLEDDRCECVAL\
QDFRAGEQIYIFYGTRSNAEFVIHSGFFFDNNSHDRVKIKLGVSKSDRLYAMKAEVLARA\
GIPTSSVFALHFTEPPISAQLLAFLRVFCMTEEELKEHLLGDSAIDRIFTLGNSEFPVSW\
DNEVKLWTFLEDRASLLLKTYKTTIEEDKSVLKNHDLSVRAKMAIKLRLGEKEILEKAVK\
SAAVNREYYRQQMEEKAPLPKYEESNLGLLESSVGDSRLPLVLRNLEEEAGVQDALNIRE\
AISKAKATENGLVNGENSIPNGTRSENESLNQESKRAVEDAKGSSSDSTAGVKE', 'SETD3', 'Actin-histidine N-methyltransferase']
proteins.loc[10] = ['MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQS\
KRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMT\
QIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDL\
AGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSY\
ELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLS\
GGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQ\
EYDESGPSIVHRKCF', 'ACTB', 'cytoplasmic 1']
proteins.loc[11] = ['MEEEIAALVIDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQS\
KRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMT\
QIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDL\
AGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSY\
ELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLS\
GGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQ\
EYDESGPSIVHRKCF', 'ACTG1', 'cytoplasmic 2']
proteins.loc[12] = ['MTSPCSPPLKPPISPPKTPVPQASSIPSPPLPPSPLDFSALPSPPWSQQTPVPPPLPLPP\
PPAATGPAPRHVFGLEKSQLLKEAFDKAGPVPKGREDVKRLLKLHKDRFRGDLRWILFCA\
DLPSLIQEGPQCGLVALWMAGTLLSPPSGVPLERLIRVATERGYTAQGEMFSVADMGRLA\
QEVLGCQAKLLSGGLGGPNRDLVLQHLVTGHPLLIPYDEDFNHEPCQRKGHKAHWAVSAG\
VLLGVRAVPSLGYTEDPELPGLFHPVLGTPCQPPSLPEEGSPGAVYLLSKQGKSWHYQLW\
DYDQVRESNLQLTDFSPSRATDGRVYVVPVGGVRAGLCGQALLLTPQDCSH', 'ACTMAP', 'Actin maturation protease']

# Add Histones proteins in human body
proteins.loc[13] = ['MDISTRSKDPGSAERTAQKRKFPSPPHSSNGHSPQDTSTSPIKKKKKPGLLNSNNKEQSE\
LRHGPFYYMKQPLTTDPVDVVPQDGRNDFYCWVCHREGQVLCCELCPRVYHAKCLRLTSE\
PEGDWFCPECEKITVAECIETQSKAMTMLTIEQLSYLLKFAIQKMKQPGTDAFQKPVPLE\
QHPDYAEYIFHPMDLCTLEKNAKKKMYGCTEAFLADAKWILHNCIIYNGGNHKLTQIAKV\
VIKICEHEMNEIEVCPECYLAACQKRDNWFCEPCSNPHPLVWAKLKGFPFWPAKALRDKD\
GQVDARFFGQHDRAWVPINNCYLMSKEIPFSVKKTKSIFNSAMQEMEVYVENIRRKFGVF\
NYSPFRTPYTPNSQYQMLLDPTNPSAGTAKIDKQEKVKLNFDMTASPKILMSKPVLSGGT\
GRRISLSDMPRSPMSTNSSVHTGSDVEQDAEKKATSSHFSASEESMDFLDKSTASPASTK\
TGQAGSLSGSPKPFSPQLSAPITTKTDKTSTTGSILNLNLDRSKAEMDLKELSESVQQQS\
TPVPLISPKRQIRSRFQLNLDKTIESCKAQLGINEISEDVYTAVEHSDSEDSEKSDSSDS\
EYISDDEQKSKNEPEDTEDKEGCQMDKEPSAVKKKPKPTNPVEIKEELKSTSPASEKADP\
GAVKDKASPEPEKDFSEKAKPSPHPIKDKLKGKDETDSPTVHLGLDSDSESELVIDLGED\
HSGREGRKNKKEPKEPSPKQDVVGKTPPSTTVGSHSPPETPVLTRSSAQTSAAGATATTS\
TSSTVTVTAPAPAATGSPVKKQRPLLPKETAPAVQRVVWNSSSKFQTSSQKWHMQKMQRQ\
QQQQQQQNQQQQPQSSQGTRYQTRQAVKAVQQKEITQSPSTSTITLVTSTQSSPLVTSSG\
SMSTLVSSVNADLPIATASADVAADIAKYTSKMMDAIKGTMTEIYNDLSKNTTGSTIAEI\
RRLRIEIEKLQWLHQQELSEMKHNLELTMAEMRQSLEQERDRLIAEVKKQLELEKQQAVD\
ETKKKQWCANCKKEAIFYCCWNTSYCDYPCQQAHWPEHMKSCTQSATAPQQEADAEVNTE\
TLNKSSQGSSSSTQSAPSETASASKEKETSAEKSKESGSTLDLSGSRETPSSILLGSNQG\
SDHSRSNKSSWSSSDEKRGSTRSDHNTSTSTKSLLPKESRLDTFWD', 'ZMYND8', 'MYND-type zinc finger-containing chromatin reader ZMYND8']
proteins.loc[14] = ['MAESSESFTMASSPAQRRRGNDPLTSSPGRSSRRTDALTSSPGRDLPPFEDESEGLLGTE\
GPLEEEEDGEELIGDGMERDYRAIPELDAYEAEGLALDDEDVEELTASQREAAERAMRQR\
DREAGRGLGRMRRGLLYDSDEEDEERPARKRRQVERATEDGEEDEEMIESIENLEDLKGH\
SVREWVSMAGPRLEIHHRFKNFLRTHVDSHGHNVFKERISDMCKENRESLVVNYEDLAAR\
EHVLAYFLPEAPAELLQIFDEAALEVVLAMYPKYDRITNHIHVRISHLPLVEELRSLRQL\
HLNQLIRTSGVVTSCTGVLPQLSMVKYNCNKCNFVLGPFCQSQNQEVKPGSCPECQSAGP\
FEVNMEETIYQNYQRIRIQESPGKVAAGRLPRSKDAILLADLVDSCKPGDEIELTGIYHN\
NYDGSLNTANGFPVFATVILANHVAKKDNKVAVGELTDEDVKMITSLSKDQQIGEKIFAS\
IAPSIYGHEDIKRGLALALFGGEPKNPGGKHKVRGDINVLLCGDPGTAKSQFLKYIEKVS\
SRAIFTTGQGASAVGLTAYVQRHPVSREWTLEAGALVLADRGVCLIDEFDKMNDQDRTSI\
HEAMEQQSISISKAGIVTSLQARCTVIAAANPIGGRYDPSLTFSENVDLTEPIISRFDIL\
CVVRDTVDPVQDEMLARFVVGSHVRHHPSNKEEEGLANGSAAEPAMPNTYGVEPLPQEVL\
KKYIIYAKERVHPKLNQMDQDKVAKMYSDLRKESMATGSIPITVRHIESMIRMAEAHARI\
HLRDYVIEDDVNMAIRVMLESFIDTQKFSVMRSMRKTFARYLSFRRDNNELLLFILKQLV\
AEQVTYQRNRFGAQQDTIEVPEKDLVDKARQINIHNLSAFYDSELFRMNKFSHDLKRKMI\
LQQF', 'MCM2', 'DNA replication licensing factor MCM2']
proteins.loc[15] = ['MAKVQVNNVVVLDNPSPFYNPFQFEITFECIEDLSEDLEWKIIYVGSAESEEYDQVLDSV\
LVGPVPAGRHMFVFQADAPNPGLIPDADAVGVTVVLITCTYRGQEFIRVGYYVNNEYTET\
ELRENPPVKPDFSKLQRNILASNPRVTRFHINWEDNTEKLEDAESSNPNLQSLLSTDALP\
SASKGWSTSENSLNVMLESHMDCM', 'ASF1A', 'Histone chaperone ASF1A']
proteins.loc[16] = ['MSGGFELQPRDGGPRVALAPGETVIGRGPLLGITDKRVSRRHAILEVAGGQLRIKPIHTN\
PCFYQSSEKSQLLPLKPNLWCYLNPGDSFSLLVDKYIFRILSIPSEVEMQCTLRNSQVLD\
EDNILNETPKSPVINLPHETTGASQLEGSTEIAKTQMTPTNSVSFLGENRDCNKQQPILA\
ERKRILPTWMLAEHLSDQNLSVPAISGGNVIQGSGKEEICKDKSQLNTTQQGRRQLISSG\
SSENTSAEQDTGEECKNTDQEESTISSKEMPQSFSAITLSNTEMNNIKTNAQRNKLPIEE\
LGKVSKHKIATKRTPHKEDEAMSCSENCSSAQGDSLQDESQGSHSESSSNPSNPETLHAK\
ATDSVLQGSEGNKVKRTSCMYGANCYRKNPVHFQHFSHPGDSDYGGVQIVGQDETDDRPE\
CPYGPSCYRKNPQHKIEYRHNTLPVRNVLDEDNDNVGQPNEYDLNDSFLDDEEEDYEPTD\
EDSDWEPGKEDEEKEDVEELLKEAKRFMKRK', 'APLF', 'Aprataxin and PNK-like factor']
proteins.loc[17] = ['MAKVSVLNVAVLENPSPFHSPFRFEISFECSEALADDLEWKIIYVGSAESEEFDQILDSV\
LVGPVPAGRHMFVFQADAPNPSLIPETDAVGVTVVLITCTYHGQEFIRVGYYVNNEYLNP\
ELRENPPMKPDFSQLQRNILASNPRVTRFHINWDNNMDRLEAIETQDPSLGCGLPLNCTP\
IKGLGLPGCIPGLLPENSMDCI', 'ASF1B', 'Histone chaperone ASF1B']
proteins.loc[18] = ['MGLLDLCEEVFGTADLYRVLGVRREASDGEVRRGYHKVSLQVHPDRVGEGDKEDATRRFQ\
ILGKVYSVLSDREQRAVYDEQGTVDEDSPVLTQDRDWEAYWRLLFKKISLEDIQAFEKTY\
KGSEEELADIKQAYLDFKGDMDQIMESVLCVQYTEEPRIRNIIQQAIDAGEVPSYNAFVK\
ESKQKMNARKRRAQEEAKEAEMSRKELGLDEGVDSLKAAIQSRQKDRQKEMDNFLAQMEA\
KYCKSSKGGGKKSALKKEKK', 'DNAJC9', 'DnaJ homolog subfamily C member 9']

def get_seq(aa_col):
    '''
    function to get the sequence of amino acids which proteins in human body are made by
    '''
    aa_seq = ''
    for each_aa in aa_col:
        aa_seq += each_aa
    return aa_seq

def vocab(sequence):
    '''
    function to build the vocabulary
    '''
    chars = sorted(list(set(sequence))) # unique character in the sequence
    vocab_size = len(chars) # the length of unique characters
    return chars, vocab_size

def encoder(chars, sequence):
    '''
    character-level function to encode unique character in the sequence
    '''    
    chtoi = {char: ind for ind, char in enumerate(chars)} # character to integer-for encoder part
    encode = lambda text: [chtoi[char] for char in text] # encoder function
    tokens = torch.tensor(encode(sequence), dtype = torch.int64)
    return tokens 

def decoder(chars, tokens):
    '''
    character-level function to decode integeres to characters
    '''
    itoch = {ind:char for ind, char in enumerate(chars)} # integer to character-for decoder part
    sequence = lambda en_list: ''.join(itoch[integer] for integer in en_list) # decoder function
    return sequence

def split(tokens):
    '''
    function to split data into train and validation set
    '''
    train_len = int(len(tokens)*0.9)
    train_set = tokens[:train_len] #90% of data -> train set
    val_set = tokens[train_len:] #10% of data -> validation set
    return train_set, val_set

def get_data(n):
    s_token = np.array([2]) 
    e_token = np.array([3])
    '''
    function to get the data, train and validation set
    '''
    aa_col = proteins['amino acid']
    sequence = get_seq(aa_col)
    chars, vocab_size = vocab(sequence) 
    tokens = encoder(chars, sequence)
    train_set, val_set = split(tokens)

    train_data = train_set
    val_data = val_set
    length = 8
    data = []
    
    for i in range(n // 3):
        X = np.concatenate((s_token, np.ones(length), e_token))
        y = np.concatenate((s_token, np.ones(length), s_token))
        data.append([X, y])
    for i in range(n // 3):
        X = np.concatenate((s_token, np.zeros(length), e_token))
        y = np.concatenate((s_token, np.zeros(length), e_token))
        data.append([X, y])
    for i in range(n // 3):
        X = np.ones(length)
        start = random.randint(0, 1)
        X[start::2] = 1
        y = np.zeros(length)
        if X[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1
        X = np.concatenate((s_token, X, e_token))
        y = np.concatenate((s_token, y, e_token))
        data.append([X, y])

    return data

# block_size is the length of sequence that we train the model with
# batch is the number of block_sized data that are processed in parallel 
def get_batch(data, batch_size=16, padding=False, padding_token=-1):
    '''
    batch is the number of block_sized data that are processed in parallel 
    '''
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches