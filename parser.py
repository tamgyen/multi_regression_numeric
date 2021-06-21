import pandas as pd
import numpy as np
from progressbar import progressbar

targetPath = "C:/Dev/Projects/turbine/data/Achilles_gene_effect.csv"
expPath = "C:/Dev/Projects/turbine/data/CCLE_expression.csv"
cnPath = "C:/Dev/Projects/turbine/data/CCLE_gene_cn.csv"

def readCsv(datPath):
    print("Reading csv: " + datPath)
    df = pd.read_csv(datPath)
    print("Loaded dataframe of shape [" + str(len(df.index)) + ", " + str(len(df.columns)) + "]\n")

    return df


def parseTarget(targ, colNames, doShift, addLogicMat, doScale):
    print("Parsing target data" + targetPath)
    targFilt = pd.DataFrame(data=targ['DepMap_ID'])
    logicMat = np.zeros((len(targ.index), len(colNames)), dtype=float)
    for col in progressbar(targ.columns, prefix="parsing: "):
        name = col.split(' ')[0]
        for i, gene in enumerate(colNames):
            if name == gene:
                targCol = targ[col]
                if addLogicMat:
                    for j in range(0, len(targCol.index)):
                        if targCol.iloc[j] > -.5:
                            logicMat[j, i] = 1
                if doShift:
                    targCol = targCol + 0.5
                if doScale:
                    colMin = targCol.min()
                    colMax = targCol.max()
                    targColScaled = (targCol-colMin)/colMax-colMin

                targFilt = pd.concat([targFilt, targCol], axis=1)
    if addLogicMat:
        targFilt = pd.concat([targFilt, pd.DataFrame(data=logicMat)], axis=1)
    print("parsed target!\n")
    return targFilt


def filterByIndex(dfRef, dfsamp):
    print("Cross-filtering for matched lines..")
    dfRefFilt = dfRef.copy()
    dfRefFilt.set_index(dfRef.iloc[:, 0], inplace=True)
    dfFilt = pd.DataFrame(columns=dfsamp.columns)
    dfsamp.set_index(dfsamp.iloc[:, 0], inplace=True)
    for iref in progressbar(dfRef.iloc[:, 0], prefix="filtering: "):
        m = False
        for isamp in dfsamp.iloc[:, 0]:
            if iref == isamp:
                dfFilt = dfFilt.append(pd.DataFrame(data=dfsamp.loc[[iref]], columns=dfsamp.columns))
                m = True
                # print("matched " + iref)
        if not m:
            # print("no match for " + iref + "..removing")
            dfRefFilt = dfRefFilt.drop(index=[iref])
    print("matched: " + str(len(dfFilt.index)) + "\nnot found: " + str(len(dfRef.index) - len(dfRefFilt.index)) + "\n")
    return dfFilt, dfRefFilt


targ = readCsv(targetPath)
exp = readCsv(expPath)
cn = readCsv(cnPath)

targetsParsed = parseTarget(targ, colNames=['ABCA1', 'CDK1', 'ERBB2', 'SOS1'], doShift=True,  addLogicMat=True)
targetsParsed.to_csv("./data/" + "targets_parsed.csv", sep=',', index=None, mode='a', float_format='%.14f')

expFilt, targetsFilt = filterByIndex(targetsParsed, exp)
cnFilt, targetsFilt = filterByIndex(targetsFilt, cn)

dsOut = pd.concat([targetsFilt, expFilt.iloc[:,1:], cnFilt.iloc[:,1:]], axis=1)
dsOutExpOnly = pd.concat([targetsFilt, expFilt.iloc[:,1:]], axis=1)
dsOutCnOnly = pd.concat([targetsFilt, cnFilt.iloc[:,1:]], axis=1)

dsOut.to_csv("./data/" + "ds.csv", sep=',', index=None, mode='a', float_format='%.14f')
dsOutExpOnly.to_csv("./data/" + "ds_exp.csv", sep=',', index=None, mode='a', float_format='%.14f')
dsOutCnOnly.to_csv("./data/" + "ds_cn.csv", sep=',', index=None, mode='a', float_format='%.14f')






print(":)")