import mhe.HorizonExtractor3;
import mhe.LocalSlopeFinder;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Вызов оригинальных классов Xinming Wu без изменения их формул.
 *
 * Параметры по умолчанию скопированы из mhe/src/mhe/demo3.py,
 * пример f3d в том же репозитории:
 *   LocalSlopeFinder(8, 2, 2, 5)
 *   planarity^8
 *   setWeights(0), setSmoothings(8, 8), setCG(0.01, 100), setExternalIterations(20)
 *   surfaceUpdateFromSlopesAndCorrelations(5, 20, ...)
 *
 * Оси: n1 = depth, n2 = inline, n3 = crossline.
 * CSV: horizon_id,depth,crossline,inline  (координаты нашего куба).
 * В метод Wu уходят k1=depth, k2=inline, k3=crossline.
 */
public final class ExtractHorizons {
  private static final class Point {
    final float depth, crossline, inline;
    Point(float d, float c, float i) { depth = d; crossline = c; inline = i; }
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 4) {
      System.err.println("Usage: java ExtractHorizons seismic.bin points.csv output-dir meta.json [outerIterations]");
      System.exit(2);
    }
    Locale.setDefault(Locale.US);
    Path bin = Paths.get(args[0]);
    Path csv = Paths.get(args[1]);
    Path outDir = Paths.get(args[2]);
    Path meta = Paths.get(args[3]);
    int outer = args.length > 4 ? Integer.parseInt(args[4]) : 20;
    Files.createDirectories(outDir);

    int n1 = 0, n2 = 0, n3 = 0;
    for (String line : Files.readAllLines(meta)) {
      line = line.replace(" ", "");
      if (line.contains("\"n1_depth\"")) n1 = number(line);
      if (line.contains("\"n2_inline\"")) n2 = number(line);
      if (line.contains("\"n3_crossline\"")) n3 = number(line);
    }
    if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
      throw new IOException("В volume_meta.json нет n1_depth / n2_inline / n3_crossline");
    }
    System.out.println("n1 depth=" + n1 + "  n2 inline=" + n2 + "  n3 crossline=" + n3);

    float[][][] fx = readVolume(bin, n1, n2, n3);
    Map<String, List<Point>> groups = readCsv(csv, n1, n2, n3);

    // Наклоны считаются один раз на весь куб, как slopes() в demo3.py.
    float[][][] p2 = new float[n3][n2][n1];
    float[][][] p3 = new float[n3][n2][n1];
    float[][][] ep = new float[n3][n2][n1];
    LocalSlopeFinder slopes = new LocalSlopeFinder(8.0, 2.0, 2.0, 5.0);
    System.out.println("Считаю наклоны LocalSlopeFinder(8, 2, 2, 5), как в demo3.py");
    slopes.findSlopes(fx, p2, p3, ep);
    float[][][] wp = new float[n3][n2][n1];
    for (int i3 = 0; i3 < n3; ++i3)
      for (int i2 = 0; i2 < n2; ++i2)
        for (int i1 = 0; i1 < n1; ++i1)
          wp[i3][i2][i1] = (float) Math.pow(Math.max(0.0, ep[i3][i2][i1]), 8.0);

    List<String> ids = new ArrayList<String>(groups.keySet());
    float[][][] surfaces = new float[ids.size()][][];
    for (int h = 0; h < ids.size(); ++h) {
      String id = ids.get(h);
      List<Point> pts = groups.get(id);
      float[] k1 = new float[pts.size()];
      float[] k2 = new float[pts.size()];
      float[] k3 = new float[pts.size()];
      for (int i = 0; i < pts.size(); ++i) {
        k1[i] = pts.get(i).depth;
        k2[i] = pts.get(i).inline;      // n2
        k3[i] = pts.get(i).crossline;   // n3
      }
      HorizonExtractor3 extractor = new HorizonExtractor3();
      extractor.setWeights(0.0f);
      extractor.setSmoothings(8.0f, 8.0f);
      extractor.setCG(0.01f, 100);
      extractor.setExternalIterations(outer);
      System.out.println("Горизонт " + id + ", точек " + pts.size());
      float[][] initial = extractor.surfaceInitialization(n1, n2, n3, k1, k2, k3);
      float[][] surface = extractor.surfaceUpdateFromSlopesAndCorrelations(
          5, 20, fx, wp, p2, p3, k2, k3, initial);
      surfaces[h] = surface;
      double err = 0.0;
      for (int i = 0; i < pts.size(); ++i) {
        int ic = Math.round(k3[i]);
        int ii = Math.round(k2[i]);
        err += Math.abs(surface[ic][ii] - k1[i]);
      }
      System.out.println("  средняя ошибка на контрольных точках, отсчётов: " + (err / pts.size()));
    }

    // Сохраняем (горизонт, crossline, inline), float32 little-endian, как ждёт numpy.
    Path raw = outDir.resolve("horizons_raw.bin");
    try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(raw)))) {
      for (int h = 0; h < surfaces.length; ++h)
        for (int c = 0; c < n3; ++c)
          for (int i = 0; i < n2; ++i)
            writeLE(o, surfaces[h][c][i]);
    }
    StringBuilder names = new StringBuilder();
    for (int h = 0; h < ids.size(); ++h) {
      if (h > 0) names.append(",");
      names.append(ids.get(h));
    }
    Files.write(outDir.resolve("horizon_ids.txt"), names.toString().getBytes("UTF-8"));
    System.out.println("Готово: " + raw);
  }

  private static int number(String line) {
    int colon = line.indexOf(':');
    int end = line.indexOf(',', colon);
    if (end < 0) end = line.indexOf('}', colon);
    return Integer.parseInt(line.substring(colon + 1, end).trim());
  }

  private static void writeLE(DataOutputStream o, float v) throws IOException {
    int bits = Float.floatToIntBits(v);
    o.writeByte(bits & 0xff);
    o.writeByte((bits >> 8) & 0xff);
    o.writeByte((bits >> 16) & 0xff);
    o.writeByte((bits >> 24) & 0xff);
  }

  private static float[][][] readVolume(Path path, int n1, int n2, int n3) throws IOException {
    long expected = 4L * n1 * n2 * n3;
    if (Files.size(path) != expected) {
      throw new IOException("Размер " + path + " = " + Files.size(path) + ", ожидалось " + expected);
    }
    float[][][] f = new float[n3][n2][n1];
    try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
      final int chunk = 1 << 28; // 256 MB, лимит одного MappedByteBuffer на части JVM
      long pos = 0;
      int i3 = 0, i2 = 0, i1 = 0;
      while (pos < expected) {
        long len = Math.min(chunk, expected - pos);
        MappedByteBuffer bb = ch.map(FileChannel.MapMode.READ_ONLY, pos, len);
        bb.order(ByteOrder.BIG_ENDIAN);
        int floats = (int) (len / 4);
        for (int k = 0; k < floats; ++k) {
          f[i3][i2][i1] = bb.getFloat();
          i1++;
          if (i1 == n1) { i1 = 0; i2++; }
          if (i2 == n2) { i2 = 0; i3++; }
        }
        pos += len;
      }
    }
    return f;
  }

  private static Map<String, List<Point>> readCsv(Path csv, int n1, int n2, int n3) throws IOException {
    Map<String, List<Point>> groups = new LinkedHashMap<String, List<Point>>();
    int lineNo = 0;
    try (BufferedReader br = Files.newBufferedReader(csv)) {
      String line;
      while ((line = br.readLine()) != null) {
        lineNo++;
        line = line.trim();
        if (line.isEmpty() || line.startsWith("#")) continue;
        String[] a = line.split(",");
        if (lineNo == 1 && a[0].trim().equalsIgnoreCase("horizon_id")) continue;
        if (a.length < 4) throw new IOException("Строка " + lineNo + ": " + line);
        String id = a[0].trim();
        float d = Float.parseFloat(a[1].trim());
        float c = Float.parseFloat(a[2].trim());
        float i = Float.parseFloat(a[3].trim());
        if (!(d >= 0 && d < n1 && c >= 0 && c < n3 && i >= 0 && i < n2)) {
          throw new IOException("Точка вне куба, строка " + lineNo + ": " + line);
        }
        List<Point> list = groups.get(id);
        if (list == null) { list = new ArrayList<Point>(); groups.put(id, list); }
        list.add(new Point(d, c, i));
      }
    }
    if (groups.isEmpty()) throw new IOException("В CSV нет точек");
    for (Map.Entry<String, List<Point>> e : groups.entrySet()) {
      if (e.getValue().size() < 2) {
        System.out.println("Внимание: у " + e.getKey() + " меньше двух точек. "
            + "В demo3 одна точка даёт плоский старт, для нашего куба лучше несколько точек на разных разрезах.");
      }
    }
    return groups;
  }
}
