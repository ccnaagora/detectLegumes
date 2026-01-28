'''
conversion d'un json correspondant aux paniers
le json peut ressembler à :
{
  "panier": [
    { "label": 0, "mode": "poids", "valeur": 4 },
    { "label": 1, "mode": "poids", "valeur": 2 }
  ]
}
'''
import json
json_str = '''
{
  "panier": [
    { "label": 0, "mode": "poids", "valeur": 4 },
    { "label": 1, "mode": "poids", "valeur": 2 }
  ]
}
'''
data = json.loads(json_str)

from dataclasses import dataclass

@dataclass
class Contenu:
    label: int
    mode: str
    valeur: float
    def getValeur(self  ):
        return self.valeur

leg = [Contenu(**item) for item in data["panier"]]
for c in leg:
    print(c.getValeur())
print(leg)

print(data)
print(data["panier"][0])
print(data["panier"][0]["mode"])
