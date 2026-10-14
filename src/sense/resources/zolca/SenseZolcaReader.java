import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/** Internal pipe protocol. Python captures it privately and writes the SENSE cache.
 * No database records or exception messages should be printed to a user terminal.
 */
public final class SenseZolcaReader {
    private final PrintWriter output = new PrintWriter(
        new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8)));

    static String decoded(String value) {
        return new String(Base64.getDecoder().decode(value), StandardCharsets.UTF_8);
    }

    private static String encoded(String value) {
        return Base64.getEncoder().encodeToString(
            (value == null ? "" : value).getBytes(StandardCharsets.UTF_8));
    }

    private void rows(Connection connection, String tag, String query, Set<Integer> text) throws SQLException {
        try (Statement statement = connection.createStatement(); ResultSet result = statement.executeQuery(query)) {
            int columns = result.getMetaData().getColumnCount();
            while (result.next()) {
                StringBuilder row = new StringBuilder(tag);
                for (int column = 1; column <= columns; column++) {
                    row.append('\t').append(text.contains(column)
                        ? encoded(result.getString(column)) : Long.toString(result.getLong(column)));
                }
                output.println(row);
            }
        }
    }

    private Map<Long, String> referenceProducts(Connection connection) throws SQLException {
        Map<Long, Long> owners = new HashMap<>();
        Map<Long, String> flowNames = new HashMap<>();
        Map<Long, String> products = new HashMap<>();
        try (Statement statement = connection.createStatement(); ResultSet result = statement.executeQuery(
            "SELECT id,f_quantitative_reference FROM tbl_processes")) {
            while (result.next()) {
                long owner = result.getLong(1);
                long reference = result.getLong(2);
                if (!result.wasNull() && reference != 0) {
                    if (owners.put(reference, owner) != null) {
                        throw new SQLException("Duplicate reference exchange");
                    }
                }
            }
        }
        try (Statement statement = connection.createStatement(); ResultSet result = statement.executeQuery(
            "SELECT id,name FROM tbl_flows WHERE flow_type <> 'ELEMENTARY_FLOW'")) {
            while (result.next()) flowNames.put(result.getLong(1), result.getString(2));
        }
        // A sequential heap scan avoids thousands of random seeks in a compressed
        // multi-gigabyte exchange table. Only quantitative-reference flow names leave
        // this loop. No technosphere amounts or supply-chain links are exported.
        try (Statement statement = connection.createStatement(); ResultSet result = statement.executeQuery(
            "SELECT id,f_flow FROM tbl_exchanges -- DERBY-PROPERTIES index=null\n")) {
            while (result.next()) {
                Long owner = owners.get(result.getLong(1));
                if (owner == null) continue;
                String product = flowNames.get(result.getLong(2));
                if (product == null) throw new SQLException("Reference product is missing");
                products.put(owner, product);
            }
        }
        return products;
    }

    private void read(Connection connection) throws SQLException {
        if (!connection.getMetaData().isReadOnly()) throw new SQLException("Archive is not read-only");
        output.println("BEGIN\t1");
        output.flush();
        Map<Long, String> products = referenceProducts(connection);
        rows(connection, "C", "SELECT id,f_category,name FROM tbl_categories", Set.of(3));
        rows(connection, "L", "SELECT id,code FROM tbl_locations", Set.of(2));
        try (Statement statement = connection.createStatement(); ResultSet result = statement.executeQuery(
            "SELECT id,ref_id,f_location,library FROM tbl_processes")) {
            while (result.next()) {
                long owner = result.getLong(1);
                output.println("P\t" + owner + "\t" + encoded(result.getString(2)) + "\t"
                    + result.getLong(3) + "\t" + encoded(products.get(owner)) + "\t" + encoded(result.getString(4)));
            }
        }
        rows(connection, "U", "SELECT id,name FROM tbl_units", Set.of(2));
        rows(connection, "G", "SELECT id,f_reference_unit FROM tbl_unit_groups", Set.of());
        rows(connection, "Q", "SELECT id,f_unit_group FROM tbl_flow_properties", Set.of());
        rows(connection, "F", "SELECT name,f_category,f_reference_flow_property FROM tbl_flows WHERE flow_type='ELEMENTARY_FLOW'", Set.of(1));
        rows(connection, "I", "SELECT id,name,f_category,f_activity_unit FROM tbl_social_indicators", Set.of(2));
        output.println("ASSESSMENTS_BEGIN");
        output.flush();
        long count = 0;
        try (Statement statement = connection.createStatement(); ResultSet result = statement.executeQuery(
            "SELECT f_process,f_indicator,activity_value,risk_level FROM tbl_social_aspects ORDER BY f_process,id")) {
            while (result.next()) {
                long owner = result.getLong(1);
                long indicator = result.getLong(2);
                double value = result.getDouble(3);
                String number = result.wasNull() ? "nan" : Double.toHexString(value);
                output.println("A\t" + owner + "\t" + indicator + "\t" + number + "\t" + encoded(result.getString(4)));
                count++;
            }
        }
        output.println("END\t" + count);
        output.flush();
        if (output.checkError()) throw new SQLException("Output pipe closed");
    }

    public static void main(String[] args) {
        // Every filesystem argument arrives as ASCII Base64. This avoids the
        // Windows JVM launcher's legacy-code-page conversion of Unicode paths.
        int status = 0;
        ClassLoader originalLoader = Thread.currentThread().getContextClassLoader();
        try {
            if (args.length != 3) throw new IllegalArgumentException("Invalid reader arguments");
            Path temporary = Path.of(decoded(args[1]));
            System.setProperty("derby.stream.error.file", temporary.resolve("derby.log").toString());
            System.setProperty("derby.storage.tempDirectory", temporary.resolve("derby-temp").toString());
            if (!args[2].equals("read")) throw new IllegalArgumentException("Unknown reader mode");
            try (ArchiveResources resources = new ArchiveResources(Path.of(decoded(args[0])))) {
                Thread.currentThread().setContextClassLoader(resources);
                try (Connection connection = DriverManager.getConnection("jdbc:derby:classpath:/")) {
                    new SenseZolcaReader().read(connection);
                }
            }
        } catch (Exception error) {
            // SQL exception messages may contain proprietary labels. Never emit them.
            System.err.println("SENSE native database read failed (details suppressed).");
            status = 1;
        } finally {
            Thread.currentThread().setContextClassLoader(originalLoader);
        }
        System.exit(status);
    }
}
