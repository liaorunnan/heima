class ChineseSkill:
    def __init__(self,caodai):
        self.caodai = caodai

        self.caodai_dict = {
            "唐":"静夜思"  ,
            "宋":"满江红" ,
        }

    def execute(self):
        return self.caodai_dict[self.caodai]
        
        
        
        