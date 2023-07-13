from lib.spec.spec import Task
from lib.synthesizer.top_level_synthesizer import TopLevelSynthesizer

synthesizer = TopLevelSynthesizer()

tasks = [
     Task(['4', '5+1', '0'], ['0-9', '1+11'], {}),
    Task(['US78409V1044', 'CA82509L1076'], ['BMG5876H1051', 'IE00BLP1HW54'], {}),
]

results = []
for task in tasks:
    res = synthesizer.synthesize(task)
    results.append(res)
    print("SYNTHESIZED RESULTS:", res)
