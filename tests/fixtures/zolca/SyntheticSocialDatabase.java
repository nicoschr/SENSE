import java.nio.file.*;
import java.sql.*;
import java.util.*;
import java.util.zip.*;

/** Entirely invented data. No commercial database is read or required. */
public class SyntheticSocialDatabase {
    public static void main(String[] args) throws Exception {
        Path root=Paths.get(new String(Base64.getDecoder().decode(args[0]), java.nio.charset.StandardCharsets.UTF_8));
        System.setProperty("derby.stream.error.file",root.resolve("fixture-derby.log").toString());
        Files.createDirectories(root);
        Path db=root.resolve("synthetic-db");
        String url="jdbc:derby:"+db.toAbsolutePath();
        try(Connection c=DriverManager.getConnection(url+";create=true"); Statement s=c.createStatement()) {
            String[] sql={
                "CREATE TABLE tbl_categories (id BIGINT PRIMARY KEY,f_category BIGINT,name VARCHAR(2048))",
                "CREATE TABLE tbl_locations (id BIGINT PRIMARY KEY,code VARCHAR(16))",
                "CREATE TABLE tbl_processes (id BIGINT PRIMARY KEY,ref_id VARCHAR(36),f_location BIGINT,name VARCHAR(2048),library VARCHAR(255),f_quantitative_reference BIGINT)",
                "CREATE TABLE tbl_units (id BIGINT PRIMARY KEY,name VARCHAR(64))",
                "CREATE TABLE tbl_unit_groups (id BIGINT PRIMARY KEY,f_reference_unit BIGINT)",
                "CREATE TABLE tbl_flow_properties (id BIGINT PRIMARY KEY,f_unit_group BIGINT)",
                "CREATE TABLE tbl_flows (id BIGINT PRIMARY KEY,name VARCHAR(2048),f_category BIGINT,f_reference_flow_property BIGINT,flow_type VARCHAR(64))",
                "CREATE TABLE tbl_social_indicators (id BIGINT PRIMARY KEY,name VARCHAR(2048),f_category BIGINT,f_activity_unit BIGINT)",
                "CREATE TABLE tbl_social_aspects (id BIGINT PRIMARY KEY,f_process BIGINT,f_indicator BIGINT,activity_value DOUBLE,risk_level VARCHAR(64))",
                "CREATE TABLE tbl_exchanges (id BIGINT PRIMARY KEY,f_flow BIGINT)",
                "CREATE TABLE tbl_unrelated_matrix (id BIGINT PRIMARY KEY,payload VARCHAR(4096))",
                "INSERT INTO tbl_categories VALUES (1,NULL,'Invented workers'),(2,1,'Synthetic safety and training')",
                "INSERT INTO tbl_locations VALUES (1,'AA'),(2,'BB')",
                "INSERT INTO tbl_processes VALUES (101,'00000000-0000-4000-8000-000000000101',1,'Invented café process','',501),(102,'00000000-0000-4000-8000-000000000102',2,'Invented 工場 process','',502),(103,'00000000-0000-4000-8000-000000000103',1,'Invented process without assessments','',503)",
                "INSERT INTO tbl_units VALUES (1,'h')",
                "INSERT INTO tbl_unit_groups VALUES (1,1)",
                "INSERT INTO tbl_flow_properties VALUES (1,1)",
                "INSERT INTO tbl_flows VALUES (41,'Invented product A',2,1,'PRODUCT_FLOW'),(42,'Invented product B',2,1,'PRODUCT_FLOW'),(43,'Invented product C',2,1,'PRODUCT_FLOW'),(44,'Synthetic safety; high risk',2,1,'ELEMENTARY_FLOW'),(45,'Synthetic training; high opportunity',2,1,'ELEMENTARY_FLOW'),(46,'Synthetic safety; no data',2,1,'ELEMENTARY_FLOW'),(47,'Synthetic training; low opportunity',2,1,'ELEMENTARY_FLOW')",
                "INSERT INTO tbl_social_indicators VALUES (201,'Synthetic safety',2,1),(202,'Synthetic training',2,1)",
                "INSERT INTO tbl_social_aspects VALUES (1,101,201,2.0,'HIGH_RISK'),(2,101,202,0.125,'HIGH_OPPORTUNITY'),(3,101,202,0.125,'HIGH_OPPORTUNITY'),(4,102,201,0.0,'NO_DATA'),(5,102,202,-1.25,'LOW_OPPORTUNITY'),(6,102,201,NULL,'NO_DATA')",
                "INSERT INTO tbl_exchanges VALUES (501,41),(502,42),(503,43)"
            };
            for(String statement:sql)s.execute(statement);
            if(args.length>1 && args[1].equals("invalid-risk"))s.execute("UPDATE tbl_social_aspects SET risk_level='UNSUPPORTED_RISK' WHERE id=1");
            if(args.length>1 && args[1].equals("library"))s.execute("UPDATE tbl_processes SET library='invented-library' WHERE id=101");
            try(PreparedStatement p=c.prepareStatement("INSERT INTO tbl_unrelated_matrix VALUES (?,?)")) {
                for(int i=0;i<20;i++){p.setInt(1,i);p.setString(2,"invented".repeat(500));p.executeUpdate();}
            }
        }
        try {DriverManager.getConnection(url+";shutdown=true");}
        catch(SQLException e){if(!"08006".equals(e.getSQLState()))throw e;}
        try(ZipOutputStream out=new ZipOutputStream(Files.newOutputStream(root.resolve("synthetic.zolca"))); java.util.stream.Stream<Path> files=Files.walk(db)) {
            Iterator<Path> iterator=files.iterator();
            while(iterator.hasNext()) {
                Path file=iterator.next();
                if(!Files.isRegularFile(file))continue;
                out.putNextEntry(new ZipEntry(db.relativize(file).toString().replace('\\','/')));
                Files.copy(file,out);out.closeEntry();
            }
        }
    }
}
