import numpy as np 
import scipy as scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)


class Sudokuuuuu(object):
    def __init__(self, soudoku):
        self.soudoku = np.zeros((9,9))
        self.given = []
        self.translator(soudoku)
        self.soft_solution = np.zeros((9,9,9))
        self.hard_solution = np.zeros((9,9))
        self.lam = None
        
    def translator(self, sudoku):
        D = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
        
        assert(len(sudoku)==81)
        for t in range(81):
            i = t//9
            j = t - i*9
            if sudoku[t] in D.keys():
                # print(sudoku[t])
                # print(i)
                # print(j)
                self.soudoku[i,j] = D[sudoku[t]]
        
        
        for i in range(9):
            for j in range(9):
                if self.soudoku[i,j]>0:
                    self.given.append([i, j, int(self.soudoku[i,j])])



        

    def non_convex_admm_solver(self, X_init=None, lam_init = None, total_it = 20000, rho = 5., acc = 1e-20):

        I = np.eye(9);

        # %% init
        if X_init:
            X0 = X_init;
        else:
            X0 = np.zeros((9,9,9,))#X0 = np.random.randn(9,9,9)#

        for g in self.given:
            # print(g)
            # print(I[:,g[2]])
            X0[:, g[0], g[1]] = I[:,g[2]-1];


        X1 = X0;
        X2 = X0;
        X3 = X0;
        X4 = X0;
        X5 = X0;
        X6 = X0;

        if lam_init:
            lam1 = lam_init[:,:,:,0];
            lam2 = lam_init[:,:,:,1];
            lam3 = lam_init[:,:,:,2];
            lam4 = lam_init[:,:,:,3];
            lam5 = lam_init[:,:,:,4];
            lam6 = lam_init[:,:,:,5];
        else:
            lam1 = np.zeros((9,9,9));
            lam2 = np.zeros((9,9,9));
            lam3 = np.zeros((9,9,9));
            lam4 = np.zeros((9,9,9));
            lam5 = np.zeros((9,9,9));
            lam6 = np.zeros((9,9,9));

        dual1 = [];
        dual2 = [];
        dual3 = [];
        dual4 = [];
        dual5 = [];
        dual6 = [];
        dual = [];
        primal = [];


        

        # for it in range(total_it):
        for it in tqdm(range(total_it)):
            
            
            
            
            # %% updating X1 (0<=X1<=1)
            X1 = X0-(1/rho)*lam1;
            X1 = np.maximum(0,X1);
            X1 = np.minimum(1,X1);
            # %% updating X2 (row constraint)
            X2 = X0-(1/rho)*lam2;
            for i in range(9):
                for k in range(9):
                    s = np.sum(X2[k, i, :]);
                    X2[k, i, :] = (1-s)/9 + X2[k, i, :];

            # %% updating X3 (col constraint)
            X3 = X0-(1/rho)*lam3;
            for j in range(9):
                for k in range(9):
                    s = np.sum(X3[k, :, j]);
                    X3[k, :, j] = (1-s)/9 + X3[k, :, j];

            # %% updating X4 (Box constraints)
            X4 = X4-(1/rho)*lam4;
            for i in range(0,9,3):
                for j in range(0,9,3):
                    for k in range(9):
                        s = np.sum(X4[k, i:i+3, j:j+3]);
                        X4[k, i:i+3, j:j+3] = (1-s)/9 + X4[k, i:i+3, j:j+3];

            # %% updating X5 (cell constraint)
            X5 = X0-(1/rho)*lam5; 
            for i in range(9):
                for j in range(9):
                    s = np.sum(X5[:, i, j]);
                    X5[:, i, j] = (1-s)/9 + X5[:, i, j];


            # %% updating X6 (given constraints)
            X6 = X0-(1/rho)*lam6;
            
            for g in self.given:
                X6[:, g[0], g[1]] = I[:,g[2]-1];
            
            
            # %% update X0
            X0p = X0;
        #     X0 = (1/6)*(X1+X2+X3+X4+X5+X6)+(1/(6*rho))*(lam1+lam2+lam3+lam4+lam5+lam6);
            uu = 2;
            vv = 1;
            X0 = (rho/(6*rho-uu))*(X1+X2+X3+X4+X5+X6)+(1/(6*rho-uu))*(lam1+lam2+lam3+lam4+lam5+lam6-vv);
            # X0 = X0 + 0.001*np.random.randn(9,9,9)
            
            primal.append(rho*np.sum((X0p-X0)**2));

            # %% Update dual variables
            lam1 = lam1 + rho*(X1-X0);
            lam2 = lam2 + rho*(X2-X0);
            lam3 = lam3 + rho*(X3-X0);
            lam4 = lam4 + rho*(X4-X0);
            lam5 = lam5 + rho*(X5-X0);
            lam6 = lam6 + rho*(X6-X0);
            
            # %% Computing the duals
            dual1.append(np.sum((X1-X0)**2));
            dual2.append(np.sum((X2-X0)**2));
            dual3.append(np.sum((X3-X0)**2));
            dual4.append(np.sum((X4-X0)**2));
            dual5.append(np.sum((X5-X0)**2));
            dual6.append(np.sum((X6-X0)**2));
            dual.append(rho*(dual1[-1]+dual2[-1]+dual3[-1]+dual4[-1]+dual5[-1]+dual6[-1]));
            mmm = dual[-1] + primal[-1]
            if mmm<acc:
                break

        lam = np.concatenate((lam1[:,np.newaxis], lam2[:,np.newaxis], lam3[:,np.newaxis], lam4[:,np.newaxis], lam5[:,np.newaxis], lam6[:,np.newaxis]), axis=3)
        plt.semilogy(dual)
        plt.ylabel('dual')
        plt.show()

        plt.semilogy(primal)
        plt.ylabel('primal')
        plt.show()

        return X0, lam, dual, primal

    def simple_round(self, X0):
        T = np.zeros((9,9));
        X_f = np.zeros((9,9,9));
        I = np.eye(9);
        for i in range(9):
            for j in range(9):
                T[i,j] = int(np.argmax(X0[:,i,j])+1);
                X_f[:,i,j] = I[:,int(T[i,j])-1];

        return T, X_f

    def Verify(self, X):
        Row = np.zeros((9));
        Col = np.zeros((9));
        Block = np.zeros((9));
        row_sum = np.sum(X, 2);
        for i in range(9):
            for k in range(9):
                if row_sum[k,i] !=1:
                    Row[i]=1;

        col_sum = np.sum(X, 1);
        # print(col_sum.shape)
        for j in range(9):
            for k in range(9):
                if col_sum[k, j] != 1:
                    Col[j]=1;

        mis_match = np.zeros((9,9))
        
        I = np.eye(9)
        for g in self.given:
            if np.sum((X[:, g[0], g[1]] - I[:,g[2]-1])**2)>0:
                mis_match[i,j] = 1


        t = -1;
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                t = t + 1;
                for k in range(9):
                    if np.sum(X[k, i:i+3, j:j+3]) != 1:
                        Block[t] = 1;
        verification = False
        if np.sum(Row)+np.sum(Col) + np.sum(Block)+ np.sum(mis_match) == 0:
            verification = True
        return Row, Col, Block, mis_match, verification
    def OneShot(self):
        T, hard_solution = self.simple_round(self.soft_solution)
        
        row,col,block, mis_match, verification = self.Verify(hard_solution)
        if verification:
            self.hard_solution = hard_solution
            self.T = T
        return T, hard_solution, verification, row, col, block, mis_match
    
        
        







def main():
    # very deep
    sudoku = '1.......7.2.4...6...3...5...9..82.......9..8....6..4....5...1...6.8...2.7....3...'
    # sudoku = '1.......7.2.4...6...3...5...9..82.......9..8....6..4....5...1...6.8...2.7....3...'
    # Hard
    # sudoku = '1....7.9..3..2...8..96..5....53..9...1..8...26....4...3......1..4......7..7...3..' # Done! (rho = 5, init =0, total_it ~ 3000, uu = 2)
    # sudoku = '..3.......5.4...8.1.......7.9..8........94.6.5.62.............3.6.9...4...7.2.1..'
    # sudoku = '...2....871......9..6.9.5....8.6...5.....23...4....6....9.3..5.4....1........7...' # Done! (rho = 1000, init =0, total_it ~ 50000, uu = 2) [soft solution is not very good with rho = 5 and total_it = 5000, but does the job]
    # sudoku = '........7.2.4...6.1.....5...9...2.4....8..6..6..9.......5..3....3..8..2.7....4..1' # Done! (rho = 5, init =0, total_it ~ 5500, uu = 2)
    # sudoku = '5......8......5.63..3.8.9....8.5...9...2.....1....4.....9.6..3.2.........4.1..7..' # tried rho=5, 100
    # sudoku = '.....1..87.......9..3.9.5....9.6...5.7...23..1..4..6....6.3..5..........2....4...' # Done! (rho = 1000, init =0, total_it ~ 32000, uu = 2) gives a solution which is wrong [solved it by removing the first clue, i.e. 1, and running with the same set up at ~20000]
    # sudoku = '........6..3.8..9..4.....85..5.9...8.7...2...1..4.......6.3...92....73.....1.....' # Done! (rho = 5, init =0, total_it ~ 4000, uu = 2)
    # sudoku = '.....2..67..1....9..3.9.5....8.6...5.....43.....2..8....6.8..5.4.........7...1...' # (rho = 1000, init =0, total_it ~ 57000, uu = 2) gives a solution which is wrong
    # sudoku = '.....4..67..1....9..3.9.5....8.3...5...2..3.....4..8....9.8..5.2....1....7.......' # tried rho=5, 100, 1000 from init 0
    # sudoku = '.......39.8......5..9.6.8....5.9...67....2......4.......3.8..5..2.7..6..4.....1..'
    # sudoku = '.......93.8......5..3.6.8....5.9...67....2......4.......9.8..5..2.7..6..4.....1..'
    # sudoku = '.2.4..7.........32.......94.9.2...7...6..5...8...1....5.1..8....3.9....7......6..'
    # sudoku = '1.......2....1..3...5..34....2..1..4....8.7..6..9.......1..5.4.8.....5..9...6....' # 

    ###########
    # sudoku = '........87.......9..3.9.5....9.6...5.7...23..1..4..6....6.3..5..........2....4...'
    # sudoku = '........87.......9..3.9.5....9.6...5.7...23.41..4..6....6.3..5..........2....4...'
    ###########
    # sudoku = '......1..4....67.9.6.21.......5.3.1..58.7...3...6.....9...8......24...5.7..3....2'
    # sudoku = '.5.4..1...7...5.48.8...7...3.65....9.........5....36.4...7...8.71.6...2...8..4.3.'
    # sudoku = '.82....6991...2........5...14..5......82.67......8..93...3........4...7623....81.'
    ###########

    # sudoku = '..7..89...9..4....6.89.............5.8.4..7......82.1..6.7.45..8.5.3....97....6..'
    # sudoku = '........1.....2.3.....3.42...42..56...6..4....1.8.......5..63...8.7....99..1.....'  # [last one on website] Done! (rho = 1000, init =0, total_it ~ 18000, uu = 2)
    # sudoku = '........1.....2.3.....3.42...42..56...63.4....1.86......5..63...8.7....99..1.....' # [another version of the previous one] (with random starting point with seed 0 after ~39K iterations find a wrong solution)

    # Medium
    # sudoku = '....8.7.2.6..129......4..8.6.5.....4.9.......2..7..5.11.....6.9..25.....78.......' # Done! (rho = 5, init =0, total_it ~ 1500, uu = 2)
    # sudoku = '..4.2.....9..8.....17..5..24..23.....58.7..2...2.....7.25.9.3..74.............81.' # Done! (rho = 5, init =0, total_it ~ 2700, uu = 2)
    # sudoku = '....5..6.7.9..1....4....8...5...3.91.........21.7...3...5....7....2..4.3.8..6....' # Done! (rho = 5, init =0, total_it ~ 2700, uu = 2)


    S = Sudokuuuuu(sudoku)
    S.soft_solution, S.lam, _, _ = S.non_convex_admm_solver(rho=1000., total_it = 100000)
    T, hard_solution, verification, row, col, block, mis_match = S.OneShot()
    print(T)
    print('row:')
    print(row)
    print('col:')
    print(col)
    print('block:')
    print(block)
    print('mis_match:')
    print(mis_match)
    print(verification)  
    
    


    





if __name__ == "__main__":
    main()