class EnglishSkill:
    def __init__(self,caodai):
        self.caodai = caodai

        self.caodai_dict = {
            "yuan":"The Nightingale"  ,
            "qing":"Farewell to the Fire" ,
        }

    def execute(self):
        return self.caodai_dict[self.caodai]
        
        
        
        