class ModuleSelector:
    def select(self,complexity,uncertainty):
        modules = ['reasoning']
        if complexity > 0.5:
            modules.append('deep_reasoning')
        if uncertainty > 0.3:
            modules.append("self_evaluation")
        return modules