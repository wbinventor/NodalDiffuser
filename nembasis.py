
class NEMBasis:

    ############################################################################
    #############################  NEM BASIS FUNCTIONS #########################
    ############################################################################
        
    def P0(self, xi):
        return 1.


    def P1(self, xi):
        return 2. * xi - 1.


    def P2(self, xi):
        return 6. * xi * (1. - xi) - 1.

    
    def P3(self, xi):
        return 6. * xi * (1. - xi) * (2. * xi - 1.)


    def P4(self, xi):
        return 6. * xi * (1. - xi) * (5. * xi**2 - 5. * xi + 1.)


    ############################################################################
    ###################  FIRST DERIVATIVES OF NEM BASIS FUNCTIONS  #############
    ############################################################################

    def DP0(self, xi):
        0.


    def DP1(self, xi):
        return 2.


    def DP2(self, xi):
        return 6. * (1. - xi) - 6. * xi


    def DP3(self, xi):
        return 12. * (1. - xi) * xi + 6. * (1. - xi) * (-1. + 2. * xi) - \
               6. * xi * (-1. + 2. * xi)


    def DP4(self, xi):
        return 6. * (1. - xi) * xi * (-5. + 10. * xi) + 6. * (1. - xi) * \
               (1. - 5. * xi + 5. * xi**2) - 6. * xi * (1. - 5. * xi + 5.*xi**2)
