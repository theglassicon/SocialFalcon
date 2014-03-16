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

//#define TOTAL_FEATURES   5
#define MAX_EPOCHS       100
#define MIN_EPOCHS       10                  // Minimum number of epochs per feature
#define MIN_IMPROVEMENT  0.00001              // Minimum improvement required to continue current feature

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


double LRATE1u  =            0.01;        // Learning rate parameter for features
double LAMDA1u  =            0.015;        // reg for features
double LRATE1m  =            0.01;        // Learning rate parameter for features
double LAMDA1m  =            0.015;        // reg for features


double LRATE2ub =            0.01;        // Learning rate parameter for biases
double LAMDA2ub =            0.01;        // reg for biases
double LRATE2mb =            0.01;        // Learning rate parameter for biases
double LAMDA2mb =            0.01;          // reg for biases

double dP =            	1.0; //0.2; //1.0;          // radius
double ksi =		0.8; //0.999; //0.8;           // constraint factor 

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


double randn(void);

void array_min(double a[], double *min, int *position);

void double_array_fill_zeros (double *my_array, unsigned int size_of_my_array);

double predict_svd_rating (int movieId, int custId, int TOTAL_FEATURES);

int rnd(int max);


void create_txt_file ();
void write_txt_file (unsigned int customer_id, unsigned short movie_id);
void close_txt_file ();



void calc_features(int TOTAL_FEATURES);

double sigmoid (double alpha);


FILE *lgfile=NULL;
void lg(char *fmt,...);
void lgopen(int argc, char**argv);
void error(char *fmt,...);

double final_probe_rmse=0.0;
int final_epochs_for_probe;


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
double GLOBAL_AVERAGE;
double GLOBAL_SCALED_AVERAGE;
int min_r,max_r, rating_range;
double avg;


// ****** SVD *********** //
double **movie_features;     // Array of features by movie (using doubles to save space)
double **cust_features;   // Array of features by customer (using doubles to save space)
double *m_bias;
double *c_bias;

int **user_movies;
int *user_movies_size;
int **user_ratings;

double **D_movie_features;     // Array of features by movie (using doubles to save space)
double **D_cust_features;   // Array of features by customer (using doubles to save space)
double *D_m_bias;
double *D_c_bias;

double **movie_features_gradients;     // Array of gradients for features by movie (using doubles to save space)
double **cust_features_gradients;   // Array of gradients for features by customer (using doubles to save space)
double *m_bias_gradients;
double *c_bias_gradients;

double **F_movie_features_gradients;     // Array of gradients for features by movie (using doubles to save space)
double **F_cust_features_gradients;   // Array of gradients for features by customer (using doubles to save space)
double *F_m_bias_gradients;
double *F_c_bias_gradients;

//double diff_uv[TOTAL_FEATURES];

int *user_connections_size;
int **user_connections;

double epochs_probe_error[MAX_EPOCHS];


// ******************** //


// *** PROBE ***//
int *probe_customers;
int *probe_movies;
int *probe_real_scores;
// *** //

char algorithm_name[20];


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
  lg ("dP %f\n", dP);
  lg ("ksi %f\n", ksi);
  lg ("------------------------\n\n");

*/

  double prediction;
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

rating_range=max_r-min_r;


avg = (GLOBAL_AVERAGE - min_r) / rating_range;
GLOBAL_SCALED_AVERAGE = log(avg / (1 - avg));

//printf("Maximum rating is %d, Minimum rating is %d and Range is %d\n",max_r,min_r,rating_range);
//exit(-1);


//printf("%d %d %d %d %f\n",TOTAL_MOVIES, TOTAL_CUSTOMERS, TOTAL_RATES, TOTAL_PROBES,GLOBAL_AVERAGE);
//exit(-1);

// ****** SVD *********** //

//movie_features = ( double** )malloc(TOTAL_MOVIES*sizeof( double* ));
//cust_features = ( double** )malloc(TOTAL_CUSTOMERS*sizeof( double* ));

movie_features = ( double** )malloc(TOTAL_MOVIES * sizeof(double *));

  if(movie_features == NULL)
    {
    fprintf(stderr, "out of memory for movie features array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    movie_features[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(movie_features[i] == NULL)
      {
      fprintf(stderr, "out of memory for movie features row\n");
      exit(-1);
      }
    }

D_movie_features = ( double** )malloc(TOTAL_MOVIES * sizeof(double *));

  if(D_movie_features == NULL)
    {
    fprintf(stderr, "out of memory for movie features array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    D_movie_features[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(D_movie_features[i] == NULL)
      {
      fprintf(stderr, "out of memory for movie features row\n");
      exit(-1);
      }
    }



cust_features = ( double** )malloc(TOTAL_CUSTOMERS * sizeof(double *));

  if(cust_features == NULL)
    {
    fprintf(stderr, "out of memory for customer features array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    cust_features[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(cust_features[i] == NULL)
      {
      fprintf(stderr, "out of memory for customer features row\n");
      exit(-1);
      }
    }


D_cust_features = ( double** )malloc(TOTAL_CUSTOMERS * sizeof(double *));

  if(D_cust_features == NULL)
    {
    fprintf(stderr, "out of memory for customer features array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    D_cust_features[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(D_cust_features[i] == NULL)
      {
      fprintf(stderr, "out of memory for customer features row\n");
      exit(-1);
      }
    }


m_bias =  (double *)malloc(sizeof(double)*TOTAL_MOVIES);

D_m_bias =  (double *)malloc(sizeof(double)*TOTAL_MOVIES);


c_bias =  (double *)malloc(sizeof(double)*TOTAL_CUSTOMERS);

D_c_bias =  (double *)malloc(sizeof(double)*TOTAL_CUSTOMERS);


movie_features_gradients = ( double** )malloc(TOTAL_MOVIES * sizeof(double *));

  if(movie_features_gradients == NULL)
    {
    fprintf(stderr, "out of memory for movie gradients array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    movie_features_gradients[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(movie_features_gradients[i] == NULL)
      {
      fprintf(stderr, "out of memory for movie gradients array row\n");
      exit(-1);
      }
    }


F_movie_features_gradients = ( double** )malloc(TOTAL_MOVIES * sizeof(double *));

  if(F_movie_features_gradients == NULL)
    {
    fprintf(stderr, "out of memory for movie gradients array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    F_movie_features_gradients[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(F_movie_features_gradients[i] == NULL)
      {
      fprintf(stderr, "out of memory for movie gradients array row\n");
      exit(-1);
      }
    }


cust_features_gradients = ( double** )malloc(TOTAL_CUSTOMERS * sizeof(double *));

  if(cust_features_gradients == NULL)
    {
    fprintf(stderr, "out of memory for customer gradients array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    cust_features_gradients[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(cust_features_gradients[i] == NULL)
      {
      fprintf(stderr, "out of memory for customer gradients row\n");
      exit(-1);
      }
    }


F_cust_features_gradients = ( double** )malloc(TOTAL_CUSTOMERS * sizeof(double *));

  if(F_cust_features_gradients == NULL)
    {
    fprintf(stderr, "out of memory for customer gradients array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    F_cust_features_gradients[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(F_cust_features_gradients[i] == NULL)
      {
      fprintf(stderr, "out of memory for customer gradients row\n");
      exit(-1);
      }
    }


m_bias_gradients =  (double *)malloc(sizeof(double)*TOTAL_MOVIES);

F_m_bias_gradients =  (double *)malloc(sizeof(double)*TOTAL_MOVIES);


c_bias_gradients =  (double *)malloc(sizeof(double)*TOTAL_CUSTOMERS);

F_c_bias_gradients =  (double *)malloc(sizeof(double)*TOTAL_CUSTOMERS);



user_connections = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));

  if(user_connections == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }
  
user_connections_size =  (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);


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
  int c, d, h,f, e, i, j, custId,vcustId, vmcustId,  cnt = 0;

  int v,w;  

  int *mutual_neighbours_id;

  int num_movies;
  int num_neighbours;
  int num_mutual_neighbours;
  int num_mutual_num_neighbours;

 
  double err, err2, p, sq, rmse_last, rmse = 2.0, probe_rmse=9998, probe_rmse_last=9999, probe_sq;
       
  int movieId;
  double cf, mf, cf_bias, mf_bias;


  double Tuv;

  double cf_bias_vt;

  double dQ, lamda2, lamda1;

  double IJJ,IJF,IFF;

  double dwdw;
  double diff_uv[TOTAL_FEATURES]; //lol


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


////////////////First we count how many movies exist in our train dataset and store them

sprintf(query_string,"SELECT COUNT(DISTINCT item_id) FROM train");

res = mysql_perform_query(conn,query_string);

int num_train_movies;

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_train_movies=atoi(row[0]);
}

/* clean up the database result set */
mysql_free_result(res);

//printf("Found %d movies in the training set\n", num_train_movies);
//exit(-1);


//////////Now we select train  movies and store them in an array

int *train_movies_id;

train_movies_id = (int *)malloc(sizeof(int)*num_train_movies);

///The select query
sprintf(query_string,"SELECT DISTINCT item_id FROM train ORDER BY item_id");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      train_movies_id[h]=atoi(row[0]);
      //printf("%d %d\n",h+1, train_movies_id[h]);
      h++;
 }

/* clean up the database result set */
mysql_free_result(res);



////////////////First we count how many users exist in our train dataset and store them

sprintf(query_string,"SELECT COUNT(DISTINCT user_id) FROM train");

res = mysql_perform_query(conn,query_string);

int num_train_users;

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_train_users=atoi(row[0]);
}

/* clean up the database result set */
mysql_free_result(res);


//////////Now we select all train users and store them in an array

int *train_users_id;

train_users_id = (int *)malloc(sizeof(int)*num_train_users);

///The select query
sprintf(query_string,"SELECT DISTINCT user_id FROM train ORDER BY user_id");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      train_users_id[h]=atoi(row[0]);
      //printf("%d %d\n",h+1, train_users_id[h]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);

//exit(-1);

////////Now we have the train set users stored



//////////Now we select ALL  users and store them in an array

int *all_users_id;

all_users_id = (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);

///The select query
sprintf(query_string,"SELECT DISTINCT user_id FROM user_mapping");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      all_users_id[h]=atoi(row[0]);
      //printf("%d %d\n",h+1, all_users_id[h]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);

//exit(-1);

////////Now we have ALL users stored

//////Now we need to find all connections for all users

 for (c=0; c<TOTAL_CUSTOMERS;c++) {


custId = c+1;
//printf("CustId: %d\n", custId);

//////Find out how many movies the user have rated

sprintf(query_string,"select count(item_id) FROM train WHERE user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_movies=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);


 if (num_movies!=0) {

user_movies_size[c] = num_movies;


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



sprintf(query_string,"select count(target_user_id) from trust where source_user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_neighbours=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);

    if (num_neighbours!=0) {

user_connections_size[c]=num_neighbours;

user_connections[c] = ( int* )malloc(num_neighbours * sizeof(int));
    
    if(user_connections[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }
   


sprintf(query_string,"select target_user_id from trust where source_user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

///fetch all selected rows


  h=0; //Just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
       user_connections[c][h]=atoi(row[0]);
       h++;
      }
  
 /* clean up the database result set */
mysql_free_result(res);


///////We got all neigbours for this user
 } // if 

} // for (all users)


//exit(-1);

  // Keep looping until you have stopped making significant (probe_rmse) progress
    while ((probe_rmse < probe_rmse_last - MIN_IMPROVEMENT)) { 
    //for (e=0; e < MAX_EPOCHS; e++) {
        
    start = time(NULL);
    start_e = time(NULL);
    
    cnt++;
    sq = 0;
    probe_sq = 0;
    rmse_last = rmse;
    probe_rmse_last = probe_rmse;

// RESET all feature gradients 
  for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i<TOTAL_MOVIES; i++) {
      movie_features_gradients[i][f] = 0.0;
     // printf("%f\n",movie_features[i][f]);
    }
    for (i=0; i<TOTAL_CUSTOMERS; i++) {
      cust_features_gradients[i][f] = 0.0;
    //  printf("%f\n",cust_features[i][f]);
    }
  }


// *** RESET biases gradients
  for (i=0; i<TOTAL_MOVIES; i++) {
    m_bias_gradients[i] = 0.0;
   // printf("%f\n",m_bias[i]);
  }
  for (i=0; i<TOTAL_CUSTOMERS; i++) {
    c_bias_gradients[i] = 0.0;
    //printf("%f\n",c_bias[i]);
  }



///////continue with the train iteration


     for (c=0; c < num_train_users; c++)  {

      d=c;

      custId = train_users_id[c];

      //custId = c+1;

      //printf("Train_custId %d\n", custId);




/// READY FOR ITERATIONS ////

      if (user_movies_size[custId-1]!=0) {


      for (i=0; i< user_movies_size[custId-1]; i++) {

        //movieId=movie_id[i];

        movieId=user_movies[custId-1][i];

        //printf("%d %d \n",movieId, user_ratings[custId-1][i]);

        p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

        err2 = - ( (double)user_ratings[custId-1][i] - (rating_range*p + min_r) );

        //err2 = - ( ((double)rating[i] - min_r) / rating_range - p);

        sq += err2*err2;

        //*** train biases
        cf_bias = c_bias[custId - 1];
        mf_bias = m_bias[movieId - 1];

        c_bias_gradients[custId - 1] += err2 * rating_range * p  * (1.0 - p);

        m_bias_gradients[movieId - 1] += err2 * rating_range * p * (1.0 - p);

        for (f=0; f<TOTAL_FEATURES; f++) {      

          // Cache off old feature values
          cf = cust_features[custId - 1][f];
          mf = movie_features[movieId - 1][f];

         cust_features_gradients[custId - 1][f] += err2 * rating_range * p * (1.0 - p) * mf;

         movie_features_gradients[movieId - 1][f] += err2 * rating_range * p * (1.0 - p) * cf;

        }

     }

  }


} //Pass over TRAIN users

//exit(-1);


//exit(-1);



if (cnt==1) {

for (c=0; c < TOTAL_CUSTOMERS; c++)  {

      d=c;

      //custId = all_users_id[c];
      custId = c+1;
      //printf("User %d\n",custId);
      

cf_bias_vt=0.0;

for (f=0;f<TOTAL_FEATURES;f++) {
      diff_uv[f] = 0.0;
      }


Tuv=0.0;


//////Now we need to find all neighbours of this user

         if (user_connections_size[c]!=0) {



Tuv=1.0/user_connections_size[c];


//printf("Tuv is %lf because neighbors are %d\n",Tuv,num_neighbours);
//exit(-1);


  for (i=0;i<user_connections_size[c];i++) {

       vcustId=user_connections[c][i];
     
      for (f=0;f<TOTAL_FEATURES;f++) {
      diff_uv[f] += cust_features[vcustId - 1][f];

      //printf("%lf\n", diff_uv[f]);
      }
  

     cf_bias_vt += c_bias[vcustId - 1];    

   }


     //diffUV[f]= cust_features[custId - 1][f] - diff_uv[f]*Tuv;
     //cf_bias_v = c_bias[custId - 1] - cf_bias_vt*Tuv;

///////We got all DIFFS for this user (and its neighbors)


} else {
//printf("Found a case for %d\n", custId);
}

//SOCIAL CONSTRAINTS

        F_c_bias_gradients[custId - 1] = cf_bias_vt*Tuv;

        for (f=0; f<TOTAL_FEATURES; f++) {

        F_cust_features_gradients[custId - 1][f] = diff_uv[f]*Tuv;      
         
        }
 

     //if (custId==49288)
     //exit(-1);

      
    if ((d!=0) && (d%1000000 == 0)){

    stop = time(NULL);
    diff = difftime(stop,start);
    printf("Done %d Customers in %f secs\n",d,diff);
    start = time(NULL);

   }
  

  //free((void *)movie_id);
  //free((void *)rating);
 

//     printf("Eimai mesa!\n");          

//    for (f=0; f<TOTAL_FEATURES; f++) {
//    for (i=0; i<TOTAL_MOVIES; i++) {
//     printf("%g ",movie_features[i][f]);
//    }
//    printf("\n");
//}

} // TOTAL Customers


for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
     F_movie_features_gradients[movieId - 1][f] = movie_features[movieId - 1][f]; 
    }
}


 for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
    F_m_bias_gradients[movieId - 1] = m_bias[movieId - 1];
  }


} else {
/*
      for (f=0; f<TOTAL_FEATURES; f++) {
      for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
      F_cust_features_gradients[custId - 1][f] = D_cust_features[custId - 1][f];
    }
  }


    for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
     F_c_bias_gradients[custId - 1] = D_c_bias[custId - 1];
  }

*/

for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
     F_movie_features_gradients[movieId - 1][f] = D_movie_features[movieId - 1][f];
     //printf("FFFF %g ",F_movie_features_gradients[movieId - 1][f]);
    }
   }

for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
    F_m_bias_gradients[movieId - 1] = D_m_bias[movieId - 1];
  }

}



IJJ=0.0;
IJF=0.0;
IFF=0.0;

for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
      IJJ += pow(cust_features_gradients[custId - 1][f],2);
      IJF += cust_features_gradients[custId - 1][f] * F_cust_features_gradients[custId - 1][f];
      IFF += pow(F_cust_features_gradients[custId - 1][f],2);
    }
  }


 for (i=0; i< num_train_users; i++) {
    custId = train_users_id[i];
    IJJ += pow(c_bias_gradients[custId - 1],2);
    IJF += c_bias_gradients[custId - 1] * F_c_bias_gradients[custId - 1];
    IFF += pow(F_c_bias_gradients[custId - 1],2);
  }


for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
     IJJ += pow(movie_features_gradients[movieId - 1][f],2);
     IJF += movie_features_gradients[movieId - 1][f] * F_movie_features_gradients[movieId - 1][f];
     IFF += pow(F_movie_features_gradients[movieId - 1][f],2);
    }
  }


for (i=0; i < num_train_movies; i++) {
    movieId = train_movies_id[i];
    IJJ += pow(m_bias_gradients[movieId - 1],2);
    IJF += m_bias_gradients[movieId - 1] * F_m_bias_gradients[movieId - 1];
    IFF += pow(F_m_bias_gradients[movieId - 1],2); 
  }


dQ=-ksi*dP*sqrt(IJJ);

// ************* DEBUG *************
//rmse=0.5*sq;
//rmse = sqrt(sq/TOTAL_RATES);
//printf("dQ is %g and dE=%g because RMSE is %g and RMSE_LAST is %g\n", dQ, (rmse-rmse_last), rmse, rmse_last);


lamda2=0.5*1/sqrt(((IJJ*dP*dP)-dQ*dQ)/(IFF*IJJ-IJF*IJF));

//lamda2=0.5*(((IJJ*dP^2)-dQ^2)/(IFF*IJJ-IJF^2))^(-0.5);

lamda1=(IJF-(2*lamda2*dQ))/IJJ;


   // BATCH UPDATE FACTORS

for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
     D_movie_features[movieId - 1][f] = -((lamda1/(2*lamda2))*movie_features_gradients[movieId - 1][f]) + ((1/(2*lamda2))*F_movie_features_gradients[movieId - 1][f]);
    //printf("DDD %g ",D_movie_features[movieId - 1][f]);
   }


    for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
      D_cust_features[custId - 1][f] = -((lamda1/(2*lamda2))*cust_features_gradients[custId - 1][f])+ ((1/(2*lamda2))*F_cust_features_gradients[custId - 1][f]);
    }

 }

for (i=0; i < num_train_movies; i++) {
    movieId = train_movies_id[i];
    D_m_bias[movieId - 1] = -((lamda1/(2*lamda2))*m_bias_gradients[movieId - 1]) + ((1/(2*lamda2))*F_m_bias_gradients[movieId - 1]);
  }


  for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
    D_c_bias[custId - 1] = -((lamda1/(2*lamda2))*c_bias_gradients[custId - 1]) + ((1/(2*lamda2))*F_c_bias_gradients[custId - 1]); 
  }


// ************* DEBUG *************
/*
dwdw=0.0;

for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
     dwdw += pow(D_cust_features[custId - 1][f],2);
    }
  }


  for (i=0; i< num_train_users; i++) {
     custId = train_users_id[i];
     dwdw += pow(D_c_bias[custId - 1],2);
  }

printf("dwdw is %g and dP^2 is %g\n", dwdw,dP*dP);
*/


 for (c=0; c < TOTAL_CUSTOMERS; c++)  {

      d=c;

      //custId = all_users_id[c];
      custId = c+1;
      //printf("User %d\n",custId);
      

cf_bias_vt=0.0;

for (f=0;f<TOTAL_FEATURES;f++) {
      diff_uv[f] = 0.0;
      }



Tuv=0.0;


//////Now we need to find all neighbours of this user

         if (user_connections_size[c]!=0) {



Tuv=1.0/user_connections_size[c];


//printf("Tuv is %lf because neighbors are %d\n",Tuv,num_neighbours);
//exit(-1);


  for (i=0;i<user_connections_size[c];i++) {

       vcustId=user_connections[c][i];
     
      for (f=0;f<TOTAL_FEATURES;f++) {
      diff_uv[f] += D_cust_features[vcustId - 1][f];

      //printf("%lf\n", diff_uv[f]);
      }
  

     cf_bias_vt += D_c_bias[vcustId - 1];    

   }


     //diffUV[f]= cust_features[custId - 1][f] - diff_uv[f]*Tuv;
     //cf_bias_v = c_bias[custId - 1] - cf_bias_vt*Tuv;

///////We got all DIFFS for this user (and its neighbors)


} else {
//printf("Found a case for %d\n", custId);
}

//SOCIAL CONSTRAINTS

        F_c_bias_gradients[custId - 1] = cf_bias_vt*Tuv;

        for (f=0; f<TOTAL_FEATURES; f++) {

        F_cust_features_gradients[custId - 1][f] = diff_uv[f]*Tuv;      
         
        }
 

     //if (custId==49288)
     //exit(-1);

      
    if ((d!=0) && (d%1000000 == 0)){

    stop = time(NULL);
    diff = difftime(stop,start);
    printf("Done done %d Customers in %f secs\n",d,diff);
    start = time(NULL);

   }
  

  //free((void *)movie_id);
  //free((void *)rating);
 

//     printf("Eimai mesa!\n");          

//    for (f=0; f<TOTAL_FEATURES; f++) {
//    for (i=0; i<TOTAL_MOVIES; i++) {
//     printf("%g ",movie_features[i][f]);
//    }
//    printf("\n");
//}

} // TOTAL Customers



// ************* UPDATE *****************


for (f=0; f<TOTAL_FEATURES; f++) {

    for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
     movie_features[movieId - 1][f] += D_movie_features[movieId - 1][f]; 
    }


    for (i=0; i< num_train_users; i++) {
      custId = train_users_id[i];
      cust_features[custId - 1][f] += D_cust_features[custId - 1][f];
    }

  }


 for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
    m_bias[movieId - 1] += D_m_bias[movieId - 1];
  }

  for (i=0; i< num_train_users; i++) {
    custId = train_users_id[i];
    c_bias[custId - 1] += D_c_bias[custId - 1];
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

      p = (rating_range*p + min_r);

	//Write data to file
	fprintf(fp,"%d,%d,%d,%f\n",custId,movieId,probe_real_scores[i],p);

      err = ((double)probe_real_scores[i] - p);

      probe_sq += err*err;

    }

	//close file
	fclose(fp);


    // stop timer and display time
    stop_e = time(NULL);
    diff = difftime(stop_e, start_e);

    rmse = sqrt(sq/TOTAL_RATES);
    //rmse=0.5*sq;
    probe_rmse = sqrt(probe_sq/TOTAL_PROBES);
    
    //epochs_probe_error[e]=probe_rmse;

 //   lg("     <set x='%d' y='%g' probe='%f' /> time: %f sec\n", cnt, rmse, probe_rmse, (double) diff);

    //exit(-1);
    
   avg_diff+=diff;

  }

//    printf("\nAverage time spent in each  iteration is %f\n",  avg_diff/cnt);
 

  //array_min(epochs_probe_error, &final_probe_rmse, &final_epochs_for_probe);
  //printf("\nBest probe is %lf at Iteration %d\n",final_probe_rmse, final_epochs_for_probe);

 lg("%f\t\t%d\t\t%f sec\t\t", cnt,probe_rmse,avg_diff/cnt);
/* clean up the database link */
mysql_close(conn);


}


double predict_svd_rating (int movieId, int custId, int TOTAL_FEATURES) {

  int f;
  float sum = 0.0;

  for (f=0; f<TOTAL_FEATURES; f++) {
     sum += movie_features[movieId - 1][f] * cust_features[custId - 1][f];
     //sum += cust_features[custId - 1][f];
  }

   //printf("Sum BEFORE biases %f\n",sum);

   sum += c_bias[custId - 1] + m_bias[movieId - 1];

   //printf("Sum AFTER biases %f\n",sum);


   // *** Add residuals
   sum += GLOBAL_SCALED_AVERAGE;

   return sigmoid(sum);

}

double sigmoid (double alpha) {
  return 1.0/(1.0+exp(-alpha));
}



void lgopen(int argc, char**argv) {
	lgfile=fopen("log.txt","a");
	if(!lgfile) error("Cant open log file");
	//lg("----------------------------------------------\n");
	/* Print out the date and time in the standard format.  */
	time_t curtime=time(NULL);
	//lg("%s",ctime(&curtime));

	//int i;
	//for(i=0;i<argc;i++)
	//	lg("%s ",argv[i]);
	//lg("\n");
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

void array_min(double a[], double *min, int *position) {

*min = 9999.0;
*position=-1;

int i;

for (i=0;i<MAX_EPOCHS;i++) {
 if (a[i]<*min){
  *min=a[i];
  *position=i+1;
 }
}
}
