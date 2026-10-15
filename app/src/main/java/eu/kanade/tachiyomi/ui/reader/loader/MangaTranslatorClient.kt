package eu.kanade.tachiyomi.ui.reader.loader

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/** Client for the Kaggle manga translation service. */
@Inject
@SingleIn(AppScope::class)
class MangaTranslatorClient {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(90, TimeUnit.SECONDS)
        .build()

    /**
     * Sends one page to Kaggle and returns WebP bytes. The original bytes are always returned
     * when the URL is invalid, the request fails, or the server returns a non-image response.
     */
    fun translatePage(imageBytes: ByteArray, serverUrl: String, sourceLang: String): ByteArray {
        val normalizedUrl = serverUrl.trim().removeSuffix("/")
        if (normalizedUrl.isBlank()) return imageBytes

        return try {
            val requestBody = imageBytes.toRequestBody("image/*".toMediaType())
            val multipart = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "page.jpg", requestBody)
                .addFormDataPart("source_lang", sourceLang)
                .build()
            val request = Request.Builder()
                .url("$normalizedUrl/translate-page")
                .post(multipart)
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body
                if (!response.isSuccessful || body == null) return@use imageBytes
                val mediaType = body.contentType()?.toString().orEmpty()
                if (!mediaType.startsWith("image/")) return@use imageBytes
                body.bytes()
            }
        } catch (_: IOException) {
            imageBytes
        } catch (_: IllegalArgumentException) {
            // Covers malformed server URLs and invalid media types.
            imageBytes
        }
    }
}
