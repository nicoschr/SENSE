import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLStreamHandler;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** Read-only Derby resources with bounded caching of accessed archive blocks.
 *
 * Derby opens a fresh stream and skips to a page for each classpath read. A normal
 * ZIP stream therefore repeatedly inflates the beginning of large tables. Shared
 * cursors let forward reads continue inflating once. Recently read blocks handle
 * catalog/index revisits without expanding the database to disk.
 */
final class ArchiveResources extends ClassLoader implements AutoCloseable {
    private static final int BLOCK_BYTES = 65536;
    private static final int MAX_BLOCKS = 2048; // At most 128 MiB, across all entries.
    private final ZipFile archive;
    private final Map<String, Cursor> cursors = new HashMap<>();
    private final LinkedHashMap<String, byte[]> blocks = new LinkedHashMap<>(128, 0.75f, true) {
        private static final long serialVersionUID = 1L;
        @Override protected boolean removeEldestEntry(Map.Entry<String, byte[]> entry) {
            return size() > MAX_BLOCKS;
        }
    };

    ArchiveResources(Path path) throws IOException {
        super(ClassLoader.getSystemClassLoader());
        archive = new ZipFile(path.toFile());
    }

    @Override public URL getResource(String requested) {
        String name = requested;
        while (name.startsWith("/")) name = name.substring(1);
        ZipEntry entry = archive.getEntry(name);
        if (entry == null || entry.isDirectory()) return null;
        final String resource = name;
        try {
            // This custom URL only supplies streams, never loads archive classes.
            return URL.of(java.net.URI.create("sense:/" + resource), new URLStreamHandler() {
                @Override protected URLConnection openConnection(URL url) {
                    return new URLConnection(url) {
                        @Override public void connect() { }
                        @Override public InputStream getInputStream() {
                            return new ResourceStream(resource, entry.getSize());
                        }
                    };
                }
            });
        } catch (MalformedURLException error) {
            throw new IllegalStateException("Invalid database resource path");
        }
    }

    @Override public InputStream getResourceAsStream(String name) {
        URL resource = getResource(name);
        if (resource == null) return null;
        try { return resource.openStream(); }
        catch (IOException error) { return null; }
    }

    private synchronized byte[] block(String name, long index) throws IOException {
        String key = name + ":" + index;
        byte[] cached = blocks.get(key);
        if (cached != null) return cached;
        Cursor cursor = cursors.get(name);
        long target = index * BLOCK_BYTES;
        if (cursor == null || cursor.position > target) {
            if (cursor != null) cursor.stream.close();
            cursor = new Cursor(archive.getInputStream(archive.getEntry(name)));
            cursors.put(name, cursor);
        }
        long remaining = target - cursor.position;
        while (remaining > 0) {
            long skipped = cursor.stream.skip(remaining);
            if (skipped <= 0) {
                if (cursor.stream.read() < 0) throw new EOFException("Database page is truncated");
                skipped = 1;
            }
            remaining -= skipped;
            cursor.position += skipped;
        }
        byte[] data = cursor.stream.readNBytes(BLOCK_BYTES);
        cursor.position += data.length;
        blocks.put(key, data);
        return data;
    }

    private final class ResourceStream extends InputStream {
        private final String name;
        private final long length;
        private long position;
        private boolean closed;
        ResourceStream(String name, long length) { this.name = name; this.length = length; }
        @Override public int read() throws IOException {
            byte[] single = new byte[1];
            return read(single, 0, 1) < 0 ? -1 : single[0] & 255;
        }
        @Override public int read(byte[] destination, int offset, int count) throws IOException {
            java.util.Objects.checkFromIndexSize(offset, count, destination.length);
            if (closed) throw new IOException("Resource stream is closed");
            if (count == 0) return 0;
            if (position >= length) return -1;
            byte[] data = block(name, position / BLOCK_BYTES);
            int within = (int) (position % BLOCK_BYTES);
            int copied = Math.min(count, data.length - within);
            if (copied <= 0) throw new EOFException("Database page is truncated");
            System.arraycopy(data, within, destination, offset, copied);
            position += copied;
            return copied;
        }
        @Override public long skip(long count) throws IOException {
            if (closed) throw new IOException("Resource stream is closed");
            long skipped = Math.min(Math.max(0, count), length - position);
            position += skipped;
            return skipped;
        }
        @Override public int available() { return (int) Math.min(Integer.MAX_VALUE, length - position); }
        @Override public void close() { closed = true; }
    }

    private static final class Cursor {
        final InputStream stream;
        long position;
        Cursor(InputStream stream) { this.stream = stream; }
    }

    @Override public synchronized void close() throws IOException {
        IOException failure = null;
        for (Cursor cursor : cursors.values()) {
            try { cursor.stream.close(); } catch (IOException error) { failure = error; }
        }
        cursors.clear();
        blocks.clear();
        archive.close();
        if (failure != null) throw failure;
    }
}
