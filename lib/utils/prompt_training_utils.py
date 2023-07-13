from collections import namedtuple

# sketch_v1 is Jocelyn's version
# sketch_v2 is Xi's version
PromptExamples = namedtuple('PromptExample', ['positive', 'negative', 'sketches', 'concrete', 'regex'])

TRAINING_EXAMPLES = [
    PromptExamples(positive=["(David J. Alexander), Marc Henri Sempere and Jocelyn Bulow", "(Connie Wong), Sai Wong", "(Amin Abughosh) and Joseph Abughosh and Abeer Elafifi"],
                   negative=["Connie Wong, Sai Wong", "Amin Abughosh", "Chilli House Inc."],
                   sketches={'v1': "\({??: Person}\) ((&|and|,) {??: Person})+",
                             'v2': "\({Concept: Person}\) ((&|and|,) {Concept: Person})+"},
                   concrete="\({<Person>}\) ((&|and|,) {<Person>})+",
                   regex="\([\w .]+\)(,)?( and)? [\w .]+"),
    PromptExamples(positive=["Arugello Market Corp.", "HollyFrontier Corporation", "Iron Pan, Inc."],
                   negative=["WONG JUDITH L", "South Seattle", "Brass Instrument Lubricants"],
                   sketches={'v1': "{??: Company Name} (, Inc|{??: Corporation})?(\.)?",
                             'v2': '{Concept: Company Name} (, Inc|{Concept: Corporation})?(\.)?'},
                   concrete="{<Company>} (, Inc|{<Corporation>})?(\.)?",
                   regex="[\w .]+,? (Corp|Inc)[.]?(oration)?"),
    PromptExamples(positive=['Bistro Burger Market Street', 'Coffeeshop - 3139 Mission', "Crab Station at Fisherman's Wharf"],
                   negative=["20th Century Cafe", "ALL SEASON MARKET", "AUTO CITY BRUSHLESS CAR WASH"],
                   sketches={'v1': "{??: Restaurant} ((-|at) )?{??: Location}",
                             'v2': "{Concept:Restaurant} ((-|at) )?{Concept:Location}"},
                   concrete="{<Restaurant>} ((-|at) )?{<Location>}",
                   regex=".* (at [\w ']+|Street|\d+ [\w ']+)"),
    PromptExamples(positive=["15. Mugs & Cups | Drinkware | Google Merchandise Store", "15. Bags | Google Merchandise Store", "10. Men's Outerwear | Apparel | Google Merchandise Store"],
                   negative=["2. Women's T-Shirts | Apparel | Google Merchandise Store", "22. Water Bottles & Tumblers | Drinkware | Google Merchandise Store", "Google Women's Yoga Pants"],
                   sketches={'v1': "{??: Integer}\. {??: Product} \| ({??: Category} \|)?Google Merchandise Store",
                             'v2': "{Concept: Integer}\. {Concept: Product} \| ({Concept: Category} \|)?Google Merchandise Store"},
                   concrete="{<Integer> -> NumMatch(10, <=, 20, <=}\. {<Product>} \| ({<Category>} \|)?Google Merchandise Store",
                   regex='1\d\. .*[|].*[|]?.*'),
    PromptExamples(positive=["Gift of Robert McBratney and Company|1929", "Gift of Minic Custom Woodwork, Inc. New York|1983", "Purchase, Edward C. Moore Jr. Gift|1923"],
                   negative=["Fletcher Fund, 1941", "Gift of Emma and Jay A. Lewis|2004", "The Michael C. Rockefeller Memorial Collection, Gift of Harry M. Miller Jr., and Professor Paulo de Goes, 1965"],
                   sketches={'v1': "(Purchase, )?{??: Gift}\|{??: Date}",
                             'v2': "(Purchase, )?{Concept: Gift}\|{Concept: Date}"},
                   concrete="(Purchase, )?{<Gift>}, {<Location>}?\|{<Date> -> InYear(0,2000)}",
                   regex="(Purchase, )?(Gift of .*|.* Gift)\|1\d{3}"),
    PromptExamples(positive=["0.5 m (50 cm)", "1.55 kg (1550 g)", ".5 cm (50 mm)"],
                   negative=["0.6 m (60 cm)", "2.20 kg (2200 g)", ".8 cm (80 mm)"],
                   sketches={'v1': "{??: Float} {??: Unit} \({??: Float} {??: Unit}\)",
                             'v2': "{Concept: Float} {Concept: Unit} \({Concept: Float} {Concept: Unit}\)"},
                   concrete="{<Float>}&.*[5] {<Unit>} \({<Float>} {<Unit>}\)",
                   regex="\d*[.]\d*5 (m|cm|kg) \(\d*5\d* (cm|g|mm)\)"),
    PromptExamples(positive=["0.5 m, 50 cm", "0.05 m, 5 cm", "0.05 m, 0.5 cm"],
                   negative=["0.6 m, 60 cm", "0.05 m (5 cm)", ".8 cm, 80 mm"],
                   sketches={'v1': "{??: Float} m, {??: Float} cm",
                             'v2': "{Concept: Float} m, {Concept: Float} cm"},
                   concrete="{<Float>}&.*[5] m, {<Float>} cm",
                   regex="\d*.\d*5 m, \d*.?5\d* cm"),
    PromptExamples(positive=["Director of DevOps,R&D,54,53,53,16,63,17", "Head of People Ops,Finance & Operations,,10,10,2,4,2", "Sr. Product Manager,Product,27,9,16,4,18,10"],
                   negative=["Director of DevOps,R&D,54,a,53,16,63,17", "Product, Sr. Product Manager,27,9,16,4,18,10", "Sr. Product Manager,Product,27,9,16,4,18,10,12,13"],
                   sketches={'v1': "{??: Job},{??: Department}(,{??:Integer}){6}",
                             'v2': "{Concept: Job},{Concept: Department}(,{Concept: Integer}){6}"},
                   concrete="{<Job>},{<Department>}(,{<Integer>}?){6}",
                   regex="[\w &.]+,(R&D|Finance & Operations|Product)(,\d*){6}")
]

TRAINING_EXAMPLES_PREV = [
    PromptExamples(positive=['10.5', '12.5'], negative=['12', '-1.0', '12.0', '1.0', '12.4'], sketches={'prev': 'decimal-> {??: int}[.]{??: int}'}, concrete='([0-9]){2}[.][5]', regex=""),
    PromptExamples(positive=['MAC Address=192.167.235.19', 'MAC Address=192.167.235.19; zproduct_id=XYZ'], negative=['MAC Address=192.167.235.19;', 'MAC =Address=192.167.235.19; zproduct_id=XYZ'], sketches={'prev':'key-value-> MAC Address[=]{??: IP address}([;] zproduct_id[=]{??: string})?'}, concrete='MAC Address[=][0-9A-Za-z.]+([;] zproduct_id[=][0-9A-Za-z.]+)*', regex=""),
    PromptExamples(positive=['1@100%', '9@50%', '5@100%'], negative=['0@100%', 'a@50%', '1@abc%'], sketches={'prev': 'number-percentage-> {??: int}[@]{??: int}[%]'}, concrete='[0-9][@][0-9]{1,3}[%]', regex=""),
    PromptExamples(positive=['Hercules.Cycle', 'Herbal & Product', 'Welcome @ 123'], negative=['&Hercules', 'Colgate!()', '.Youtube', '@Incule'], sketches={'prev': 'string-> {??: organization}'}, concrete='matchEntity(ORG) & [A-Z-z0-9][A-Z-z0-9 .$&-]*', regex=""),
    PromptExamples(positive=['1.1', '1.1.3.4', '1.0', '1.1.334'], negative=['1', '1.', '1..'], sketches={'prev': 'version-> {??: int}([.]{??: int})+'}, concrete='[0-9]+([.][0-9]+)+', regex=""),
    PromptExamples(positive=['10.000,50', '10,000.50'], negative=['anything@!#--'], sketches={'prev': 'currency-> {??: int}([.]{??: int})?([,]{??: int})?'}, concrete='matchEntity(CURRENCY) & [0-9,.]+', regex=""),
    PromptExamples(positive=['ACCCGTTNNGTCCGGA3', 'ACCCGTTNNGTCCGGATTGAANNGT9', 'TTGGACCNAC0', 'ACGGTA0'], negative=['BACGGTA0', 'TTGGACCNAC', 'TTGGACCNACACCCGTTNNGTCCGGATTGAANNGTTTGGACCNACACCCGTTNNGTCCGGATTGAANNGTTTGGACCNACACCCGTTNNGTCCGGATTGAANNGT2'], sketches={'prev': 'string-> {??: DNA}[0-9]'}, concrete='[AGCT]{1,64}[0-9]', regex=""),
]

PromptExamplesEntity = namedtuple('PromptExampleEntity', ['input', 'label', 'output'])

ENTITY_TRAINING_EXAMPLES = [
    PromptExamplesEntity(input='Composite.Motors,Inc.', label='Organization', output=['[Composite.Motors]', '[Composite.Motors,Inc]', '[Composite.Motors,Inc.]']),
    PromptExamplesEntity(input='Composite.Motors,Inc.', label='Person', output=['none']),
    PromptExamplesEntity(input='Big Data Architect at Madison, WI', label='Place', output=['[Madison]', '[WI]', '[Madison, WI]']),
    PromptExamplesEntity(input='470-43" Class (42.5" Diag.)   LED   1080p', label='Integer', output=['[470]', '[43]', '[1080]']),
    PromptExamplesEntity(input='2011-03-02', label='Date', output=['[2011-03-02]']),
    PromptExamplesEntity(input='1955-10-18', label='Year', output=['[1955]']),
    PromptExamplesEntity(input='404-Stream 11.6" Laptop   Intel Celeron   2GB Memory', label='Product', output=['[Stream 11.6" Laptop]', '[Intel Celeron]']),
    PromptExamplesEntity(input='Set 2 Tea Towels I Love London', label='Item', output=['[Tea Towels]', '[Tea Towels I Love London]']),
    # PromptExamplesEntity(input='Blue Circles Design Monkey Doll', label='Toy type', output=['[Doll]']),
]

PromptExamplesEntityInfer = namedtuple('PromptExamplesEntityInfer', ['pos', 'neg', 'output'])
ENTITY_INFER_EXAMPLES = [
    PromptExamplesEntityInfer(pos=['David J. Alexander', 'Marc Henri Sempere'], neg=['Chilli House Inc.', '20th Century Cafe'], output='Person'),
    PromptExamplesEntityInfer(pos=['Arugello Market Corp.', 'Iron Pan, Inc.'], neg=['South Seattle', 'Marc Henri Sempere'], output='Company'),
    PromptExamplesEntityInfer(pos=['Bistro Burger', 'Crab Station'], neg=['20th Century Cafe', 'Marc Henri Sempere'], output='Restaurant'),
    PromptExamplesEntityInfer(pos=['Mugs & Cups', "Women's T-Shirts"], neg=['Google Merchandise Store', 'Apparel'], output='Product'),
    PromptExamplesEntityInfer(pos=['Gift of Robert McBratney and Company', 'Gift of Minic Custom Woodwork, Inc.'], neg=['The Michael C. Rockefeller Memorial Collection', 'Fletcher Fund'], output='Gift'),
    PromptExamplesEntityInfer(pos=['m', 'cm'], neg=['g', 'kg'], output='Length measurement'),
    PromptExamplesEntityInfer(pos=['Director of DevOps', 'Software Engineer'], neg=['R&D', 'Finance & Operations'], output='Job'),
]