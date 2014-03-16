///////////HOW TO COMPILE ME!!!!!///////////////////

/////run icc -xT -O3 -ip -parallel db_test.c -o db_test -I/usr/include/mysql -L/usr/lib/mysql -lmysqlclient///////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <mysql.h>

//////////Database connection functions -|- Dont touch!!!!/////////////////////////////////////
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
/////////////////////////End of Database connection functions //////////////////////////////



///////////////////////Main, Run Forest Run!!!!////////////////////////////////////////////
main(int argc, char** argv) {

int start = time(NULL);


char query_string[250]; ///This is needed to construct query strings

float percentage_split=atof(argv[1]);

char db_name[30];
strcpy(db_name,argv[2]);


printf("\n\nYou selected a percentage split value of %f\nThis is a wise choice!!\n\n",percentage_split);


/////////////////Database connection data//////////////////////////////////////////////////
  MYSQL *conn;		// the connection
  MYSQL_RES *res;	// the results
  MYSQL_ROW row;	// the results row (line by line)
 
  struct connection_details mysqlD;
  mysqlD.server = "localhost";  // where the mysql database is
  mysqlD.user = "root";		// the root user of mysql	
  mysqlD.password = "checkmate#$%"; // the password of the root user in mysql
  mysqlD.database = db_name;	// the database to pick
 
  // connect to the mysql database
  conn = mysql_connection_setup(mysqlD);

//////////////////////////End of database connection data//////////////////////////////////

///////////////////////Check if user_id and item_id have incremental numbering////////////

sprintf(query_string,"SELECT COUNT(DISTINCT user_id) FROM ratings");

res = mysql_perform_query(conn,query_string);

int ALL_USERS;

while ((row = mysql_fetch_row(res)) !=NULL) {
         ALL_USERS=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"SELECT COUNT(DISTINCT item_id) FROM ratings");

res = mysql_perform_query(conn,query_string);

int ALL_ITEMS;

while ((row = mysql_fetch_row(res)) !=NULL) {
         ALL_ITEMS=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"SELECT MAX(user_id) FROM ratings");

res = mysql_perform_query(conn,query_string);

int MAX_USER_ID;

while ((row = mysql_fetch_row(res)) !=NULL) {
         MAX_USER_ID=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"SELECT MAX(item_id) FROM ratings");

res = mysql_perform_query(conn,query_string);

int MAX_ITEM_ID;

while ((row = mysql_fetch_row(res)) !=NULL) {
         MAX_ITEM_ID=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

////Now compare values 


/////////First create train_ratings table/////////////////////////////////////////////////

/////Drop table if already exists

sprintf(query_string,"DROP TABLE IF EXISTS train");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);



////Then create the apropriate table

sprintf(query_string,"CREATE TABLE train (user_id int,item_id int,rating_value int)  CHARACTER SET=utf8");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);
////////////////////////////End of table creation//////////////////////////////////////


printf("We have %d unique users in our dataset!\n\n",ALL_USERS);



///////Construct an array containning all distinct user_ids///////////////////////////////

int *users_id;

users_id = (int *)malloc(sizeof(int)*ALL_USERS);

///The select query
sprintf(query_string,"SELECT DISTINCT user_id FROM ratings");
res = mysql_perform_query(conn,query_string);

///fetch all selected rows

int h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      users_id[h]=atoi(row[0]);
      h++;
}
//clean up the database result set
mysql_free_result(res);

///////////////////End of initial phase///////////////////////////////////////////////////



int num_user_items;




for (h=0;h<ALL_USERS;h++){

sprintf(query_string,"SELECT COUNT(item_id) FROM ratings WHERE user_id=%d",users_id[h]);
res = mysql_perform_query(conn,query_string);	



while ((row = mysql_fetch_row(res)) !=NULL) {
         num_user_items=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

int num_user_items_perc = floor(num_user_items*percentage_split);

if (num_user_items_perc == 0) num_user_items_perc = 1;
 
sprintf(query_string,"INSERT INTO train SELECT user_id, item_id, rating_value FROM ratings WHERE user_id=%d ORDER BY RAND() LIMIT %d",users_id[h],num_user_items_perc);

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);


}

sprintf(query_string,"ALTER TABLE train ADD INDEX user_id (user_id)");
res = mysql_perform_query(conn,query_string);
//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"ALTER TABLE train ADD INDEX item_id (item_id)");
res = mysql_perform_query(conn,query_string);
//clean up the database result set
mysql_free_result(res);
////////Time to calculate probe_ratings////////////////////////////////////////////////////

/////////First create probe_ratings table/////////////////////////////////////////////////

/////Drop table if already exists

sprintf(query_string,"DROP TABLE IF EXISTS probe");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);



////Then create the apropriate table

sprintf(query_string,"CREATE TABLE probe (user_id int,item_id int,rating_value int)  CHARACTER SET=utf8");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);
////////////////////////////End of table creation//////////////////////////////////////

sprintf(query_string,"INSERT INTO probe SELECT ratings.user_id, ratings.item_id, ratings.rating_value FROM ratings LEFT OUTER JOIN train ON (ratings.user_id = train.user_id and ratings.item_id = train.item_id) WHERE train.user_id is NULL");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"ALTER TABLE probe ADD INDEX user_id (user_id)");
res = mysql_perform_query(conn,query_string);
//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"ALTER TABLE probe ADD INDEX item_id (item_id)");
res = mysql_perform_query(conn,query_string);
//clean up the database result set
mysql_free_result(res);

///////////////////////////////Now create table to probe only cold start users/////////

sprintf(query_string,"CREATE TABLE probe_cold (user_id int,item_id int,rating_value int)  CHARACTER SET=utf8");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"INSERT INTO probe_cold select a.user_id, item_id, rating_value from probe as a inner join (SELECT user_id FROM ratings GROUP BY user_id HAVING (COUNT( item_id ) <5 AND COUNT( item_id ) >1 )) as b on a.user_id = b.user_id");

res = mysql_perform_query(conn,query_string);

//clean up the database result set
mysql_free_result(res);

////Add indexes

sprintf(query_string,"ALTER TABLE probe_cold ADD INDEX user_id (user_id)");
res = mysql_perform_query(conn,query_string);
//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"ALTER TABLE probe_cold ADD INDEX item_id (item_id)");
res = mysql_perform_query(conn,query_string);
//clean up the database result set
mysql_free_result(res);


////Calculate and print execution time//////////////////////////////////////

int stop = time(NULL);
int diff = difftime(stop, start);
printf("Execution time was %d sec\nYou are a slow slow monkey!!!\n\n", diff);



} ////Here closes the main function !!!!!! dont touch!!!!!!!
