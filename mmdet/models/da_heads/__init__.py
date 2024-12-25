from .DA_Head import DomainAdaptationHead
# from .DA_Head_new import DomainAdaptationHead as DomainAdaptationHead_GAN
from .UDAModel_after_SCL import UDAModel_SCL
from .UDAModel import UDAModel
from .UDAModel_Xst import UDAModel_Xst
from .UDAModel_Xss import UDAModel_Xss
from .UDAModel_Xts import UDAModel_Xts
from .UDAModel_Xtt import UDAModel_Xtt
from .UDAModel_Ost import UDAModel_Ost

# __all__ = ['DomainAdaptationHead', 'UDAModel_SCL', 'UDAModel', 'DomainAdaptationHead_GAN']
__all__ = ['DomainAdaptationHead', 'UDAModel_SCL', 'UDAModel',
           'UDAModel_Xst', 'UDAModel_Xss', 'UDAModel_Xts', 'UDAModel_Xtt',
           'UDAModel_Ost'
           ]