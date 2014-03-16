#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <mysql.h>


/* *** mode: 0 for probe *** */
/* *** mode: 1 for qual  *** */
#define PREDICTION_MODE 0

/*
#if PREDICTION_MODE==0
  #define TOTAL_RATES 99072112
  #define TOTAL_PROBES 1408395
  #define GLOBAL_AVERAGE 3.603304
#else
  #define TOTAL_RATES 100480507
  #define TOTAL_PROBES 2817131
  #define GLOBAL_AVERAGE 3.60428996442066
#endif
*/



//#define TOTAL_FEATURES   10
#define MIN_EPOCHS       10                  // Minimum number of epochs per feature
#define MAX_EPOCHS       10                  // Max epochs per feature
#define MIN_IMPROVEMENT  0.00005              // Minimum improvement required to continue current feature

#define INIT_SEED_Mb       -0.3             // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_Mb   0.20             // variance range from the INIT_SEED value
#define INIT_Mb            (INIT_SEED_Mb + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_Mb) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

#define INIT_SEED_Cb       0.0              // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_Cb   0.010            // variance range from the INIT_SEED value
#define INIT_Cb            (INIT_SEED_Cb + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_Cb) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

#define INIT_SEED_M        0.0               // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_M    0.001             // variance range from the INIT_SEED value
#define INIT_M             (INIT_SEED_M + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_M) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

#define INIT_SEED_C        0.0               // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_C    0.001             // variance range from the INIT_SEED value
#define INIT_C             (INIT_SEED_C + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_C) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 


double LRATE1u  =            0.02;        // Learning rate parameter for features
double LAMDA1u  =            0.1;        // reg for features
double LRATE1m  =            0.005;        // Learning rate parameter for features
double LAMDA1m  =            0.1;        // reg for features


double LRATE2ub =            0.02;        // Learning rate parameter for biases
double LAMDA2ub =            0.01;        // reg for biases
double LRATE2mb =            0.005;        // Learning rate parameter for biases
double LAMDA2mb =            0.01;          // reg for biases



struct connection_details
{
    char *server;
    char *user;
    char *password;
    char *database;
};
 
MYSQL* mysql_connection_setup(struct connection_details mysql_details)
{
     // first of all create a mysql instance and initialize the variables within
    MYSQL *connection = mysql_init(NULL);
 
    // connect to the database with the details attached.
    if (!mysql_real_connect(connection,mysql_details.server, mysql_details.user, mysql_details.password, mysql_details.database, 0, NULL, 0)) {
      printf("Conection error : %s\n", mysql_error(connection));
      exit(1);
    }
    return connection;
}
 
MYSQL_RES* mysql_perform_query(MYSQL *connection, char *sql_query)
{
   // send the query to the database
   if (mysql_query(connection, sql_query))
   {
      printf("MySQL query error : %s\n", mysql_error(connection));
      exit(1);
   }
 
   return mysql_use_result(connection);
}


float randn(void);

void float_array_fill_zeros (float *my_array, unsigned int size_of_my_array);

float predict_svd_rating (int movieId, int custId, int TOTAL_FEATURES);

int rnd(int max);


void create_txt_file ();
void write_txt_file (unsigned int customer_id, unsigned short movie_id);
void close_txt_file ();



void calc_features(int TOTAL_FEATURES);

FILE *lgfile=NULL;
void lg(char *fmt,...);
void lgopen(int argc, char**argv);
void error(char *fmt,...);

float final_probe_rmse=0.0;
unsigned int final_epochs_for_probe;
char algorithm_name[20];

//////////////////////////database connection///////////////////////////////////////
  char query_string[200];

  MYSQL *conn;    // the connection
  MYSQL_RES *res; // the results
  MYSQL_ROW row;  // the results row (line by line)
 
  struct connection_details mysqlD;

int TOTAL_MOVIES;
int TOTAL_CUSTOMERS;
int TOTAL_RATES;
int TOTAL_PROBES;  
float GLOBAL_AVERAGE;

// ****** SVD *********** //
float **movie_features;     // Array of features by movie (using floats to save space)
float **cust_features;   // Array of features by customer (using floats to save space)
float *m_bias;
float *c_bias;

// ******************** //

int **user_movies;
int *user_movies_size;
int **user_ratings;

// *** PROBE ***//
int *probe_customers;
int *probe_movies;
int *probe_real_scores;
// *** //

int max_r,min_r;


main (int argc, char**argv) {

  lgopen(argc,argv);
/*
  lg ("\n-----------------------\nPREDICTION MODE: %s\n-----------------------\n", (PREDICTION_MODE == 1)?"qualifying":"probe");

  
  lg ("------------------------\n");
  lg ("TOTAL_FEATURES %i\n", TOTAL_FEATURES);
  lg ("MIN_IMPROVEMENT %f\n", MIN_IMPROVEMENT);
  lg ("INIT_SEED_Mb %f\n", INIT_SEED_Mb);
  lg ("INIT_VARIANCE_Mb %f\n", INIT_VARIANCE_Mb);
  lg ("INIT_SEED_Cb %f\n", INIT_SEED_Cb);
  lg ("INIT_VARIANCE_Cb %f\n", INIT_VARIANCE_Cb);
  lg ("INIT_SEED_M %f\n", INIT_SEED_M);
  lg ("INIT_VARIANCE_M %f\n", INIT_VARIANCE_M);
  lg ("INIT_SEED_C %f\n", INIT_SEED_C);
  lg ("INIT_VARIANCE_C %f\n", INIT_VARIANCE_C);
  lg ("LRATE1u %f\n", LRATE1u);
  lg ("LAMDA1u %f\n", LAMDA1u);
  lg ("LRATE1m %f\n", LRATE1m);
  lg ("LAMDA1m %f\n", LAMDA1m);
  lg ("LRATE2ub %f\n", LRATE2ub);
  lg ("LAMDA2ub %f\n", LAMDA2ub);
  lg ("LRATE2mb %f\n", LRATE2mb);
  lg ("LAMDA2mb %f\n", LAMDA2mb);
  lg ("------------------------\n\n");
*/

  float prediction;
  unsigned int i,h;
  time_t start, stop;
  double diff;
  int TOTAL_FEATURES = atoi(argv[5]);
   /* start timer */
  start = time(NULL);  

  mysqlD.server = argv[1];  // where the mysql database is
  mysqlD.user = argv[2];   // the root user of mysql 
  mysqlD.password = argv[3]; // the password of the root user in mysql
  mysqlD.database = argv[4]; // the databse to pick
 
  // connect to the mysql database
  conn = mysql_connection_setup(mysqlD);


sprintf(query_string,"SELECT count(DISTINCT item_id) FROM ratings");

res = mysql_perform_query(conn,query_string);


while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_MOVIES=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);


sprintf(query_string,"SELECT count(DISTINCT user_id) FROM user_mapping");

res = mysql_perform_query(conn,query_string);


while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_CUSTOMERS=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);


sprintf(query_string,"SELECT count(*) FROM train");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_RATES=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);


sprintf(query_string,"SELECT count(*) FROM probe");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_PROBES=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);


sprintf(query_string,"SELECT avg(rating_value) FROM train");

res = mysql_perform_query(conn,query_string);


while ((row = mysql_fetch_row(res)) !=NULL) {
         GLOBAL_AVERAGE=atof(row[0]);
}


//clean up the database result set
mysql_free_result(res);

//printf("%d %d %d %d %f\n",TOTAL_MOVIES, TOTAL_CUSTOMERS, TOTAL_RATES, TOTAL_PROBES,GLOBAL_AVERAGE);
//exit(-1);



// Get maximum and minimum ratings from the ratings table

sprintf(query_string,"SELECT MAX(rating_value) FROM ratings");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         max_r=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);


sprintf(query_string,"SELECT MIN(rating_value) FROM ratings");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         min_r=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);


//printf("%d %d %d %d %f\n",TOTAL_MOVIES, TOTAL_CUSTOMERS, TOTAL_RATES, TOTAL_PROBES,GLOBAL_AVERAGE);
//exit(-1);

// ****** SVD *********** //

//movie_features = ( float** )malloc(TOTAL_MOVIES*sizeof( float* ));
//cust_features = ( float** )malloc(TOTAL_CUSTOMERS*sizeof( float* ));

movie_features = ( float** )malloc(TOTAL_MOVIES * sizeof(float *));

  if(movie_features == NULL)
    {
    fprintf(stderr, "out of memory\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    movie_features[i] = ( float* )malloc(TOTAL_FEATURES * sizeof(float));
    if(movie_features[i] == NULL)
      {
      fprintf(stderr, "out of memory\n");
      exit(-1);
      }
    }


cust_features = ( float** )malloc(TOTAL_CUSTOMERS * sizeof(float *));

  if(cust_features == NULL)
    {
    fprintf(stderr, "out of memory\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    cust_features[i] = ( float* )malloc(TOTAL_FEATURES * sizeof(float));
    if(cust_features[i] == NULL)
      {
      fprintf(stderr, "out of memory\n");
      exit(-1);
      }
    }

m_bias =  (float *)malloc(sizeof(float)*TOTAL_MOVIES);

c_bias =  (float *)malloc(sizeof(float)*TOTAL_CUSTOMERS);

// ***************** //


user_movies = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));


  if(user_movies == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }


user_movies_size =  (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);

user_ratings = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));


   if(user_ratings == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }



/* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
//  printf("Defined global arrays: Time elapsed is %f sec\n", diff);


// *** CREATE PROBE *** //

/* start timer */
start = time(NULL);  

probe_customers = (int *)malloc(sizeof(int)*TOTAL_PROBES);
probe_movies = (int *)malloc(sizeof(int)*TOTAL_PROBES);
probe_real_scores = (int *)malloc(sizeof(int)*TOTAL_PROBES);

sprintf(query_string,"select user_id,item_id,rating_value FROM probe");

res = mysql_perform_query(conn,query_string);

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      probe_customers[h]=atoi(row[0]);
      probe_movies[h]=atoi(row[1]);
      probe_real_scores[h]=atoi(row[2]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);

// ******************** //
  

  /* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
//  printf("Created Probe arrays: Time elapsed is %f sec\n", diff);

  // start timer
  start = time(NULL);  

  // RUN SVD
//  lg("\n\nCalculating features...\n");

  sscanf(argv[0], "./%s", algorithm_name);
  lg("%s\t\t",algorithm_name);
  calc_features(TOTAL_FEATURES);

  /* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
//  printf("\nTrained SVD in %f sec\n", diff);
  lg("%f sec\n", diff);
exit(-1);

  // *** SAVE FEATURES ***
  // lg("\n\nSaving features files...\n");
  //  save_new_features_files();


 // save_predictions();

//////save_residuals();

  // stop timer and display time 
  stop = time(NULL);
  diff = difftime(stop, start);
//  lg("\nPredictions: Time elaspsed is %f sec\n", diff);

  exit(0);
}


//****** SVD *********


void calc_features(int TOTAL_FEATURES) {

  time_t start, stop, start_e, stop_e;
  double avg_diff=0.0;
  double diff;
int c, d, h,f, e, i, custId, cnt = 0;
  
  int *movie_id;
  int *rating;

  int num_movies;

  double err, p, sq, rmse_last, rmse = 2.0, probe_rmse=9998, probe_rmse_last=9999, probe_sq;
       
  int movieId;
  double cf, mf, cf_bias, mf_bias;
 
 /* 
  unsigned int startIdx, endIdx;
  unsigned int probeStartIdx, probeEndIdx;  
*/

  // INIT all feature values 
  for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i<TOTAL_MOVIES; i++) {
      movie_features[i][f] = INIT_M;
     // printf("%f\n",movie_features[i][f]);
    }
    for (i=0; i<TOTAL_CUSTOMERS; i++) {
      cust_features[i][f] = INIT_C;
    //  printf("%f\n",cust_features[i][f]);
    }
  }


  // *** INIT biases
  for (i=0; i<TOTAL_MOVIES; i++) {
    m_bias[i] = INIT_Mb;
   // printf("%f\n",m_bias[i]);
  }
  for (i=0; i<TOTAL_CUSTOMERS; i++) {
    c_bias[i] = INIT_Cb;
    //printf("%f\n",c_bias[i]);
  }
    
    /////////////////Here starts primary iteration to users
////////////////First we count how many users exist in our train dataset and store them

sprintf(query_string,"SELECT COUNT(DISTINCT user_id) FROM train");

res = mysql_perform_query(conn,query_string);

int num_train_users;

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_train_users=atoi(row[0]);
}

//////////Now we select all train users and store them in an array

int *train_users_id;

train_users_id = (int *)malloc(sizeof(int)*num_train_users);

///The select query
sprintf(query_string,"SELECT DISTINCT(user_id) FROM train");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      train_users_id[h]=atoi(row[0]);
    //printf("%d %d\n",h+1, train_users_id[h]);
      h++;
}
//exit(-1);

////////Now we have the train set users stored

///////continue with the train iteration

     for (c=0; c < num_train_users; c++)  {

      custId = train_users_id[c];

//////Find out how many movies the user have rated

sprintf(query_string,"select count(item_id) FROM train WHERE user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_movies=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);

user_movies_size[c] = num_movies;

 if (num_movies!=0) {

user_movies[c] = ( int* )malloc(num_movies * sizeof(int));

    if(user_movies[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }

user_ratings[c] = ( int* )malloc(num_movies * sizeof(int));

    if(user_ratings[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }


/////select and store the movies and ratings the user have rated

sprintf(query_string,"select item_id, rating_value FROM train WHERE user_id=%d",custId);

//printf("%s\n",query_string);

res = mysql_perform_query(conn,query_string);

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      user_movies[c][h]=atoi(row[0]);
      user_ratings[c][h]=atoi(row[1]);
      //printf("%d %d\n",user_movies[c][h],user_ratings[c][h]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);

  }


}


///////We got all movies and ratings for all users

     // Keep looping until you have stopped making significant (probe_rmse) progress
  while ((probe_rmse < probe_rmse_last - MIN_IMPROVEMENT)) {
  // for (e=0; e < 10; e++) 

    start = time(NULL);
    start_e = time(NULL);

    cnt++;
    sq = 0;
    probe_sq = 0;
    rmse_last = rmse;
    probe_rmse_last = probe_rmse;


   for (c=0; c < num_train_users; c++)  {

      d=c;

      custId = train_users_id[c];



      if (user_movies_size[c]!=0) {


      for (i=0; i< user_movies_size[c]; i++) {
      //for (i=startIdx; i<=endIdx; i++) 

        //movieId=movie_id[i];

        movieId=user_movies[c][i];

        //printf("%d %d\n",custId, movieId);
	      //exit(-1);
        
        /*  
        movieId = nf_movies_c_index[i];
        rating = (double) nf_scores_c_index[i];
        */

        // Predict rating and calc error
       // printf("Will call predict\n");
      
        p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

       // printf("After call to predict\n");

        err = ((float)user_ratings[c][i] - p);


        //printf("%d %f\n",user_ratings[c][i],err); 

        sq += err*err;


        //*** train biases
        cf_bias = c_bias[custId - 1];
        mf_bias = m_bias[movieId - 1];

        //c_bias[custId - 1] += (LRATE2ub * (err - LAMDA1u*LAMDA2ub * cf_bias));
        //m_bias[movieId - 1] += (LRATE2mb * (err - LAMDA1m*LAMDA2mb * mf_bias));          

        c_bias[custId - 1] += (LRATE2ub * (err - LAMDA2ub * cf_bias));
        m_bias[movieId - 1] += (LRATE2mb * (err - LAMDA2mb * mf_bias));
           

        for (f=0; f<TOTAL_FEATURES; f++) {      
                
          // Cache off old feature values
          cf = cust_features[custId - 1][f];
          mf = movie_features[movieId - 1][f];

          cust_features[custId - 1][f] += (LRATE1u * (err * mf - LAMDA1u * cf));
          movie_features[movieId - 1][f] += (LRATE1m * (err * cf - LAMDA1m * mf));

        }

      // if (d>30000)
      // printf("%d %d %d\n",d+1,custId,movieId);

      }

   }

      
    if ((d!=0) && (d%1000000 == 0)){

    stop = time(NULL);
    diff = difftime(stop,start);
//    printf("Done %d Customers in %f secs\n",d,diff);
    start = time(NULL);

   }
  

}



//   printf("Calculating Probe...\n");

// Open file to store probes
	char probes_file[80];
	char str_features[20];

	sprintf(str_features,"%d",TOTAL_FEATURES);

	strcpy (probes_file, str_features);
	strcat (probes_file, "-");
	strcat (probes_file, mysqlD.database);
	strcat (probes_file, "-");
	strcat (probes_file, algorithm_name);
	strcat (probes_file, ".txt");

	FILE *fp = fopen(probes_file,"w");


    for (i=0; i < TOTAL_PROBES; i++) {

      movieId = probe_movies[i];
      custId = probe_customers[i];

      // Predict rating and calc error
      p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

	//Write data to file
	fprintf(fp,"%d,%d,%d,%f\n",custId,movieId,probe_real_scores[i],p);

      err = ((float)probe_real_scores[i] - p);
      probe_sq += err*err;

    }

	//close file
	fclose(fp);

    // stop timer and display time
    stop_e = time(NULL);
    diff = difftime(stop_e, start_e);

    rmse = sqrt(sq/TOTAL_RATES);
    probe_rmse = sqrt(probe_sq/TOTAL_PROBES);

 //   lg("     <set x='%d' y='%f' probe='%f' /> time: %f sec\n", cnt, rmse, probe_rmse, (double) diff);
    
  avg_diff+=diff;

  }

//  printf("\nAverage time spent in each  iteration is %f\n",  avg_diff/cnt);
 
  final_probe_rmse = probe_rmse;
  final_epochs_for_probe = cnt;
 lg("%f\t\t%d\t\t%f sec\t\t", cnt,probe_rmse,avg_diff/cnt);
/* clean up the database link */
mysql_close(conn);


}





float predict_svd_rating (int movieId,  int custId, int TOTAL_FEATURES) {

  int f;
  double sum = 0.0;
    
  for (f=0; f<TOTAL_FEATURES; f++) {
    sum += movie_features[movieId - 1][f] * cust_features[custId - 1][f];
  }

  // *** Add biases
  sum += c_bias[custId - 1] + m_bias[movieId - 1];
    
  // *** Add residuals
  sum += GLOBAL_AVERAGE;
  
  if (sum > max_r) sum = (float) max_r;
  if (sum < min_r) sum = (float) min_r;

  return sum;
}



void lgopen(int argc, char**argv) {
	lgfile=fopen("log.txt","a");
	if(!lgfile) error("Cant open log file");
//	lg("----------------------------------------------\n");
	/* Print out the date and time in the standard format.  */
	time_t curtime=time(NULL);
//	lg("%s",ctime(&curtime));

//	int i;
//	for(i=0;i<argc;i++)
//		lg("%s ",argv[i]);
//	lg("\n");
}

void lg(char *fmt,...) {
	char buf[2048];
	va_list ap;

	va_start(ap, fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	fprintf(stderr,"%s",buf);
	if(lgfile) {
		fprintf(lgfile,"%s",buf);
		fflush(lgfile);
	}
}


void error(char *fmt,...) {
	char buf[2048];
	va_list ap;

	va_start(ap, fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	lg("%s",buf);
	lg("\n");
	exit(1);
}


