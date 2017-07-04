/**
 * Created by Nish on 2/22/2017.
 */
import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;
import Jama.Matrix;

public class GradDesc {

    private static int DebugFlag = 0;

    public static void main(String[] args) {
        System.out.println("\n\n\t\tStart of the Program.");
        ArrayList<houses> H1;

        int ProbNo = 0;
        while(ProbNo != 1 && ProbNo != 2 && ProbNo != 3) {
            System.out.print("\n\tEnter Model No.:");
            Scanner in = new Scanner(System.in);
            ProbNo = in.nextInt();
        }
        System.out.println("\n\tYou've Selected: " + ProbNo);

        H1 = readCSV("sample.csv");

        Matrix X1 = CreateMatrixX(ProbNo,H1);
        Matrix Y1 = CreateMatrixY(H1);
        GradientDescent(ProbNo,X1,Y1);

        String choice = "A";
        while(!choice.equalsIgnoreCase("Y") && !choice.equalsIgnoreCase("N")) {
            System.out.print("\n\tContinue to Cross Validation? Y/N: ");
            Scanner in = new Scanner(System.in);
            choice = in.next();
        }
        if(choice.equalsIgnoreCase("Y"))
            CrossValidation(ProbNo,X1,Y1);

        double lambda = 10;
        choice = "A";
        while(!choice.equalsIgnoreCase("Y") && !choice.equalsIgnoreCase("N")) {
            System.out.print("\n\tContinue to Ridge Regression? Y/N: ");
            Scanner in = new Scanner(System.in);
            choice = in.next();
        }
        if(choice.equalsIgnoreCase("Y")) {
            System.out.print("\n\tEnter Lambda Value : ");
            Scanner in = new Scanner(System.in);
            lambda = in.nextDouble();
            RidgeRegression(ProbNo, X1, Y1, lambda, 0);
        }

        choice = "A";
        while(!choice.equalsIgnoreCase("Y") && !choice.equalsIgnoreCase("N")) {
            System.out.print("\n\tContinue to Cross Validation? Y/N: ");
            Scanner in = new Scanner(System.in);
            choice = in.next();
        }
        if(choice.equalsIgnoreCase("Y"))
            ModelSelection(ProbNo,X1,Y1);

        System.out.println("\n\t\t\tFinished Successfully!!");

    }// main

    private static ArrayList<houses> readCSV(String csvFile)
    {
        ArrayList<houses> H1 = new ArrayList<>();
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";
        int count = 0;

        try {

            br = new BufferedReader(new FileReader(csvFile));
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] house = line.split(cvsSplitBy);

                if(!house[0].equals("id")) {
                    houses temphouse = new houses();
                    temphouse.id = Double.parseDouble(house[0].replaceAll("^\"|\"$", ""));
                    temphouse.bedrooms = Double.parseDouble(house[3].replaceAll("^\"|\"$", ""));
                    temphouse.bathrooms = Double.parseDouble(house[4].replaceAll("^\"|\"$", ""));
                    temphouse.sqft_living = Double.parseDouble(house[5].replaceAll("^\"|\"$", ""));
                    temphouse.sqft_lot = Double.parseDouble(house[6].replaceAll("^\"|\"$", ""));
                    temphouse.price = Double.parseDouble(house[2].replaceAll("^\"|\"$", ""));

                    H1.add(temphouse);

                    count++;
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        System.out.println("\t"+ count + " House Records added.");
        return H1;
    }//ReadCSV

    private static Matrix CreateMatrixX(int ProbNo, ArrayList<houses> H1) {

        int columns = 0, rows=H1.size();

        switch (ProbNo)
        {
            case 1: columns = 3;
                break;
            case 2: columns = 5;
                break;
            case 3: columns = 5;
                break;
        }

        double[][] temparray = new double [ rows ] [ columns ];

        if(ProbNo == 1 || ProbNo == 2 )
        {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < (columns); j++) {
                    temparray[i][j] = Math.pow(H1.get(i).sqft_living, j);
                }
            }
        }
        else if (ProbNo == 3)
        {
            for (int i = 0; i < rows; i++) {

                temparray[i][0] = 1;
                temparray[i][1] = H1.get(i).bedrooms;
                temparray[i][2] = H1.get(i).bathrooms;
                temparray[i][3] = H1.get(i).sqft_living;
                temparray[i][4] = H1.get(i).sqft_lot;
            }
        }
        Matrix X1 = new Matrix(temparray);
        System.out.println("\tX Matrix Created.");
        return X1;

    }//CreateMatrixX

    private static Matrix CreateMatrixY(ArrayList<houses> H1) {

        int columns = 1, rows = H1.size();
        double[][] temparray = new double[rows][1];

        for (int i = 0; i < rows; i++) {
            temparray[i][0] = H1.get(i).price;
        }
        Matrix Y1 = new Matrix(temparray);
        System.out.println("\tY Matrix Created.");
        return Y1;
    }//CreateMatrixY

    private static Matrix GradientDescent(int ProbNo,Matrix X1,Matrix Y1) {

        System.out.println("\tGradient Descent Start.");

        double stepsize = Math.pow(10,-7),rows = X1.getRowDimension();
        double chker = 2000, TotalIterations = 20000;
        double theta[][];

        switch (ProbNo)
        {
            case 2:
            case 3:
                theta = new double[5][1];
                if(DebugFlag == 1) System.out.println();
                for(int i =0; i < 5; i++){
                    theta[i][0] = 1;
                    if(DebugFlag == 1) System.out.println("\t\tTheta " + i + " : " + theta[i][0]);
                }
                break;

            case 1:
            default:
                theta = new double[3][1];
                if(DebugFlag == 1) System.out.println();
                for(int i =0; i < 3; i++){
                    theta[i][0] = 1;
                    if(DebugFlag == 1) System.out.println("\t\tTheta " + i + " : " + theta[i][0]);
                }
                break;
        }

        Matrix Theta1 = new Matrix(theta);

        try{
            for(int iteration = 0; iteration < TotalIterations; iteration++) {


                stepsize = stepsize/(iteration+1);
                Theta1 = Theta1.minus(((X1.transpose()).times(Y1.minus(X1.times(Theta1)))).times((-2*stepsize)/X1.getRowDimension()));

                if (iteration == 0)System.out.print("\t\tProcessing." );
                if((iteration+1)%chker == 0) {
                    System.out.print(".");
                    if(DebugFlag == 1)
                    {
                        for(int i =0; i < Theta1.getRowDimension(); i++) {
                            System.out.println("\t\tTheta " + i + " : " + Theta1.get(i, 0));
                        }
                    }
                }
            }
        }
        catch(ArrayIndexOutOfBoundsException e){
            System.out.println("Gradient Descent Array Error: " + e);
        }

        System.out.println();
        for(int i =0; i < Theta1.getRowDimension(); i++){
            theta[i][0] = 1;
            System.out.println("\t\tTheta " + i + " : " + Theta1.get(i, 0));
        }
        System.out.println("\tGradient Descent End.");

        return Theta1;
    }//GradientDescent

    private static void CrossValidation(int ProbNo, Matrix X1, Matrix Y1)
    {
        double rows = X1.getRowDimension(),AvgErrorRate=0;
        int PartitionSize = (int)Math.floor(rows/10);

        try
        {
            System.out.println("\n\tCross Validation Start.");

            for(int i = 0; i < 10; i++)
            {
                int PartitionStart = PartitionSize * i, PartitionEnd = PartitionSize * (i+1);
                System.out.println("\n\nCross Validation Fold: "+ (i+1));
                System.out.println("Test Partition Start: "+ PartitionStart +"\t End: "+ PartitionEnd);

                Matrix TrainingMatrix = new Matrix((X1.getRowDimension()-(PartitionSize)),X1.getColumnDimension());
                Matrix TrainingMatrixY1 = new Matrix((X1.getRowDimension()-(PartitionSize)),1);
                Matrix TestMatrix = new Matrix(PartitionSize,X1.getColumnDimension());
                Matrix TestMatrixY1 = new Matrix(PartitionSize,1);

                if(DebugFlag == 1) System.out.println("Cross Validation Checkpoint 2."+i);

                TestMatrix.setMatrix(0, PartitionSize-1, 0, X1.getColumnDimension()-1, X1.getMatrix(PartitionStart, PartitionEnd-1, 0, X1.getColumnDimension() - 1));
                if(DebugFlag == 1) System.out.println("Test X done");
                TestMatrixY1.setMatrix(0, PartitionSize-1, 0, 0, X1.getMatrix(PartitionStart, PartitionEnd-1, 0, 0));
                if(DebugFlag == 1) System.out.println("Test Y done");

                if(DebugFlag == 1) System.out.println("Cross Validation Checkpoint 1."+i);
                if(i != 0) {
                    TrainingMatrix.setMatrix(0, PartitionStart-1, 0, X1.getColumnDimension()-1, X1.getMatrix(0, PartitionStart-1, 0, X1.getColumnDimension() - 1));
                    if(DebugFlag == 1) System.out.println("Training X first done");
                    TrainingMatrixY1.setMatrix(0, PartitionStart-1, 0, 0, Y1.getMatrix(0, PartitionStart-1, 0, 0));
                    if(DebugFlag == 1) System.out.println("Training Y first done");
                }
                if(DebugFlag == 1) System.out.println("Cross Validation Checkpoint 3."+i);

                if(i != 9)
                {
                    TrainingMatrix.setMatrix(PartitionStart+1,TrainingMatrix.getRowDimension()-1,0,X1.getColumnDimension()-1, X1.getMatrix(PartitionEnd+1,X1.getRowDimension()-1,0,X1.getColumnDimension()-1));
                    if(DebugFlag == 1) System.out.println("Training X Second done");
                    TrainingMatrixY1.setMatrix(PartitionStart+1,TrainingMatrix.getRowDimension()-1,0,0, X1.getMatrix(PartitionEnd+1,X1.getRowDimension()-1,0,0));
                    if(DebugFlag == 1) System.out.println("Training Y Second done");
                }
                if(DebugFlag == 1) System.out.println("Cross Validation Checkpoint 4."+i);

                Matrix Theta1 = GradientDescent(ProbNo,TrainingMatrix, TrainingMatrixY1);
                double ErrorRate = RSS(TestMatrix,Theta1,TestMatrixY1);
                System.out.println("Error Rate on "+(i+1)+"th fold on Cross Validation : "+ ErrorRate);
                AvgErrorRate += ErrorRate;

            }
            System.out.println("Average Error Rate on 10 fold Cross Validation : "+ (AvgErrorRate/10));

        }
        catch(ArrayIndexOutOfBoundsException e)
        {
            System.out.println("\n\n\tError e:" + e);
        }

        System.out.println("\n\tCross Validation End.");

    }//CrossValidation

    private static double RSS(Matrix X1,Matrix Theta1, Matrix Y1) {
        return (((Y1.minus(X1.times(Theta1))).transpose()).times(Y1.minus(X1.times(Theta1)))).get(0,0);
    }

    private static Matrix RidgeRegression(int ProbNo,Matrix X1,Matrix Y1, double Lambda, int Nested) {

        if(Nested == 0) System.out.println("\tRidge Regression Start.");

        double stepsize = Math.pow(10,-7),rows = X1.getRowDimension();
        double chker = 2000, TotalIterations = 20000;
        double theta[][];

        switch (ProbNo)
        {
            case 2:
            case 3:
                theta = new double[5][1];
                if(DebugFlag == 1) System.out.println();
                for(int i =0; i < 5; i++){
                    theta[i][0] = 1;
                    if(DebugFlag == 1) System.out.println("\t\tTheta " + i + " : " + theta[i][0]);
                }
                break;

            case 1:
            default:
                theta = new double[3][1];
                if(DebugFlag == 1) System.out.println();
                for(int i =0; i < 3; i++){
                    theta[i][0] = 1;
                    if(DebugFlag == 1) System.out.println("\t\tTheta " + i + " : " + theta[i][0]);
                }
                break;
        }

        Matrix Theta1 = new Matrix(theta);

        try{
            for(int iteration = 0; iteration < TotalIterations; iteration++) {

                stepsize = stepsize/(iteration+1);
                Theta1 = (Theta1.times(1-2*Lambda*stepsize)).minus(((X1.transpose()).times(Y1.minus(X1.times(Theta1)))).times((-2*stepsize)/X1.getRowDimension()));

                if(Nested == 0) {
                    if (iteration == 0) System.out.print("\t\tProcessing.");
                    if ((iteration + 1) % chker == 0) {
                        System.out.print(".");
                        if (DebugFlag == 1) {
                            for (int i = 0; i < Theta1.getRowDimension(); i++) {
                                System.out.println("\t\tTheta " + i + " : " + Theta1.get(i, 0));
                            }
                        }
                    }
                }
            }
        }
        catch(ArrayIndexOutOfBoundsException e){
            System.out.println("Ridge Regression Array Error: " + e);
        }

        if(Nested == 0) {
            for (int i = 0; i < Theta1.getRowDimension(); i++) {
                System.out.println("\t\tTheta " + i + " : " + Theta1.get(i, 0));
            }

            System.out.println("\tRidge Regression End.");
        }
        return Theta1;
    }//Ridge Regression

    private static void ModelSelection(int ProbNo, Matrix X1, Matrix Y1)
    {
        double rows = X1.getRowDimension(),AvgErrorRate=0;
        int PartitionSize = (int)Math.floor(rows/10);
        double LambdaArr[] = new double[]{0.001,0.01,0.1,1,10};
        double OpLambdas[] = new double[10];

        try
        {
            System.out.println("\n\tModel Selection Start.");

            for(int i = 0; i < 10; i++)
            {
                int PartitionStart = PartitionSize * i, PartitionEnd = PartitionSize * (i+1);
                System.out.println("\n\nModel Selection Outer Fold: "+ (i+1));
                System.out.println("\tTest Partition Start: "+ PartitionStart +"\t End: "+ PartitionEnd);
                System.out.println("\t\tTest Partition Size: "+ PartitionSize);

                Matrix TrainingMatrix = new Matrix((X1.getRowDimension()-(PartitionSize)),X1.getColumnDimension());
                Matrix TrainingMatrixY1 = new Matrix((X1.getRowDimension()-(PartitionSize)),1);
                Matrix TestMatrix = new Matrix(PartitionSize,X1.getColumnDimension());
                Matrix TestMatrixY1 = new Matrix(PartitionSize,1);

                if(DebugFlag == 1) System.out.println("Model Selection Checkpoint 2."+i);

                TestMatrix.setMatrix(0, PartitionSize-1, 0, X1.getColumnDimension()-1, X1.getMatrix(PartitionStart, PartitionEnd-1, 0, X1.getColumnDimension() - 1));
                if(DebugFlag == 1) System.out.println("Test X done");
                TestMatrixY1.setMatrix(0, PartitionSize-1, 0, 0, X1.getMatrix(PartitionStart, PartitionEnd-1, 0, 0));
                if(DebugFlag == 1) System.out.println("Test Y done");

                if(DebugFlag == 1) System.out.println("Model Selection Checkpoint 1."+i);
                if(i != 0) {
                    TrainingMatrix.setMatrix(0, PartitionStart-1, 0, X1.getColumnDimension()-1, X1.getMatrix(0, PartitionStart-1, 0, X1.getColumnDimension() - 1));
                    if(DebugFlag == 1) System.out.println("Training X first done");
                    TrainingMatrixY1.setMatrix(0, PartitionStart-1, 0, 0, Y1.getMatrix(0, PartitionStart-1, 0, 0));
                    if(DebugFlag == 1) System.out.println("Training Y first done");
                }
                if(DebugFlag == 1) System.out.println("Model Selection Checkpoint 3."+i);

                if(i != 9)
                {
                    TrainingMatrix.setMatrix(PartitionStart+1,TrainingMatrix.getRowDimension()-1,0,X1.getColumnDimension()-1, X1.getMatrix(PartitionEnd+1,X1.getRowDimension()-1,0,X1.getColumnDimension()-1));
                    if(DebugFlag == 1) System.out.println("Training X Second done");
                    TrainingMatrixY1.setMatrix(PartitionStart+1,TrainingMatrix.getRowDimension()-1,0,0, X1.getMatrix(PartitionEnd+1,X1.getRowDimension()-1,0,0));
                    if(DebugFlag == 1) System.out.println("Training Y Second done");
                }
                if(DebugFlag == 1) System.out.println("Model Selection Checkpoint 4."+i);

                double OptimalLambda = 0, LeastErr = -1;
                //Calculation part
                for(int i1 =0; i1 < LambdaArr.length;i1++ )
                {
                    System.out.println("\n\t\tFor Lambda Value: "+LambdaArr[i1]);
                    double ErrorRate = RRCrossValidation(ProbNo, TrainingMatrix, TrainingMatrixY1, LambdaArr[i1]);
                    if(LeastErr < ErrorRate)
                    {
                        LeastErr = ErrorRate;
                        OptimalLambda = LambdaArr[i1];
                    }
                }
                System.out.println("\tOptimal Lambda on "+(i+1)+"th fold on Model Selection : "+ OptimalLambda);
                OpLambdas[i] = OptimalLambda;
                AvgErrorRate += LeastErr;

            }
            System.out.println("\n\n");
            for(int i = 0; i < OpLambdas.length; i++)
                System.out.println("\tOptimal Lambda "+(i+1)+" : "+ OpLambdas[i]);
            System.out.println("\n\tAverage Error Rate on Model Selection : "+ (AvgErrorRate/10));
        }
        catch(ArrayIndexOutOfBoundsException e)
        {
            System.out.println("\n\n\tArray Error e:" + e);
        }

        System.out.println("\n\tModel Selection End.");

    }//Model Selection

    private static double RRCrossValidation(int ProbNo, Matrix X1, Matrix Y1, double Lambda)
    {
        double rows = X1.getRowDimension(),AvgErrorRate=0;
        int PartitionSize = (int)Math.floor(rows/10);

        try
        {
            System.out.print("\t\t\tProcessing");

            for(int i = 0; i < 10; i++)
            {
                System.out.print(".");

                int PartitionStart = PartitionSize * i, PartitionEnd = PartitionSize * (i+1);

                Matrix TrainingMatrix = new Matrix((X1.getRowDimension()-(PartitionSize)),X1.getColumnDimension());
                Matrix TrainingMatrixY1 = new Matrix((X1.getRowDimension()-(PartitionSize)),1);
                Matrix TestMatrix = new Matrix(PartitionSize,X1.getColumnDimension());
                Matrix TestMatrixY1 = new Matrix(PartitionSize,1);

                TestMatrix.setMatrix(0, PartitionSize-1, 0, X1.getColumnDimension()-1, X1.getMatrix(PartitionStart, PartitionEnd-1, 0, X1.getColumnDimension() - 1));
                TestMatrixY1.setMatrix(0, PartitionSize-1, 0, 0, X1.getMatrix(PartitionStart, PartitionEnd-1, 0, 0));

                if(i != 0) {
                    TrainingMatrix.setMatrix(0, PartitionStart-1, 0, X1.getColumnDimension()-1, X1.getMatrix(0, PartitionStart-1, 0, X1.getColumnDimension() - 1));
                    TrainingMatrixY1.setMatrix(0, PartitionStart-1, 0, 0, Y1.getMatrix(0, PartitionStart-1, 0, 0));
                }

                if(i != 9)
                {
                    TrainingMatrix.setMatrix(PartitionStart+1,TrainingMatrix.getRowDimension()-1,0,X1.getColumnDimension()-1, X1.getMatrix(PartitionEnd+1,X1.getRowDimension()-1,0,X1.getColumnDimension()-1));
                    TrainingMatrixY1.setMatrix(PartitionStart+1,TrainingMatrix.getRowDimension()-1,0,0, X1.getMatrix(PartitionEnd+1,X1.getRowDimension()-1,0,0));
                }

                Matrix Theta1 = RidgeRegression(ProbNo,TrainingMatrix, TrainingMatrixY1,Lambda, 1);
                double ErrorRate = RSS(TestMatrix,Theta1,TestMatrixY1);
                AvgErrorRate += ErrorRate;

            }
        }
        catch(ArrayIndexOutOfBoundsException e)
        {
            System.out.println("\n\n\tError e:" + e);
        }

        return (AvgErrorRate/10);

    }//CrossValidation



}//GradDesc
