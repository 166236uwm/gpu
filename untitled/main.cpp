#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <cmath>
#include <immintrin.h>



using namespace std;

mutex mtx;

// zad 1
int counter = 0;

// zad 2
queue<int> buffer;
condition_variable cv;
bool done = false;
const int MAX_ITEMS = 10;

// zad 3
alignas(32) float vec1[1000000];
alignas(32) float vec2[1000000];
float sum[1000000];

// zad 4
int tab1[1000];
int tab2[100];

// zad 5
const int N = 10000;
const int thread_count = 4;
float vec[N];
float result_sum[thread_count];
float result_avg[thread_count];

// zad dodatkowe
int dividers = 0;

void populate(float* vec, int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] = (rand() % 1000)/100;
    }
}

void populateint(int* vec, int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] = (rand() % 5)+1;
    }
}

void wf1(int id) {
    for (int i = 0; i < 10000; ++i) {
        mtx.lock();
        if(id % 2 == 0) {
            counter++;
        } else{
            counter--;
        }
        mtx.unlock();
    }
}

void zad1() {
    cout << "Uruchamianie watkow..." << endl;

    thread thread1(wf1, 1);
    thread thread2(wf1, 2);

    thread1.join();
    thread2.join();

    cout << "Watki zakonczyly prace. ostateczna wartosc licznika to: " << counter << endl;

}
void producer() {
    for (int i = 1; i <= MAX_ITEMS; ++i) {
        unique_lock<mutex> lock(mtx);  // Blokowanie mutexa
        buffer.push(rand()%50);
        cout << "Producent: Wyprodukowano " << i << endl;
        cv.notify_one(); // Powiadomienie konsumenta
    }

    // Ustawienie flagi zakończenia i powiadomienie konsumenta
    unique_lock<mutex> lock(mtx);
    done = true;
    cv.notify_one();
}

// Funkcja konsumenta
void consumer() {
    while (true) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [] { return !buffer.empty() || done; }); // Oczekiwanie na dane

        if (buffer.empty() && done) {
            break; // Jeśli produkcja zakończona i kolejka pusta -> zakończ działanie
        }

        int item = buffer.front();
        buffer.pop();
        cout << "Konsument: Przetworzono " << item << endl << "Wynik: " << pow(item, rand()%3) << endl;
    }
}

void zad2() {
    thread producerThread(producer);
    thread consumerThread(consumer);

    producerThread.join();
    consumerThread.join();

    cout << "Zakonczono produkcje i konsumpcje." << endl;
}

void zad3() {
    populate(&vec1[0], 1000000);
    populate(&vec2[0], 1000000);

    for (int i = 0; i < 1000000; i += 8) {
        __m256 vecA = _mm256_load_ps(&vec1[i]);
        __m256 vecB = _mm256_load_ps(&vec2[i]);

        __m256 vec_sum = _mm256_add_ps(vecA, vecB);

        _mm256_store_ps(&sum[i], vec_sum);
    }
    cout << "Zakonczono dodawanie wektorow." << endl;
    for (int i = 0; i < 8; ++i) {
        printf("%f + %f = %f\n", vec1[i], vec2[i], sum[i]);
    }
}
void suma() {
    int wynik = 0;
    for (int i = 0; i < 1000; i++) {
        wynik += tab1[i];
    }
    lock_guard<mutex> lock(mtx);
    cout << "Suma elementow: " << wynik << endl;
}
void iloczyn() {
    long long wynik = 1;
    for (int i = 0; i < 100; i++) {
        wynik *= tab2[i];
    }
    lock_guard<mutex> lock(mtx);
    cout << "Iloczyn elementow: " << wynik << endl;
}

void zad4() {
    populateint(&tab1[0], 1000);
    populateint(&tab2[0], 100);

    thread s(suma);
    thread i(iloczyn);
    s.join();
    i.join();
    cout << "Zakonczono obliczenia." << endl;
}

void calc (int start, int end, int thread_id) {
    float sum = 0;
    float avg = 0;
    for (int i = start; i < end; i++) {
        sum += vec[i];
    }

    avg = sum / (end - start);
    result_sum[thread_id] = sum;
    result_avg[thread_id] = avg;
}

void zad5() {
    populate(&vec[0], N);
    thread threads[thread_count];
    int chunk_size = N / thread_count;
    for (int i = 0; i < thread_count; i++) {
        int start = i * chunk_size;
        int end = start + chunk_size;
        threads[i] = thread(calc, start, end, i);
    }
    for (int i = 0; i < thread_count; i++) {
        threads[i].join();
    }
    float total_sum = 0;
    float total_avg = 0;
    for (int i = 0; i < thread_count; i++) {
        total_sum += result_sum[i];
        total_avg += result_avg[i];
    }
    total_avg /= thread_count;
    cout << "Suma elementow: " << total_sum << endl;
    cout << "Srednia elementow: " << total_avg << endl;
}

void check_divisibility(int start, int end, int num) {
    for (int i = start; i < end; i++) {
        if (num % i == 0) {
            lock_guard<mutex> lock(mtx);
            dividers++;
        }
    }
}

void zad_dodatkowe(int max) {
    if (max < 2) {
        cout << "Brak liczb pierwszych w podanym zakresie." << endl;
        return;
    }
    for (int i = 2; i <= max; ++i) {
        dividers = 0;
        if (i % 2 != 0 || i == 2){
            int sqrt_n =static_cast<int>(sqrt(i));
            int chunk_size = sqrt_n / thread_count;
            thread threads[thread_count];
            for (int j = 0; j < thread_count; ++j) {
                int start = j * chunk_size + 1;
                int end = (j == thread_count - 1) ? (sqrt_n + 1) : (start + chunk_size);
                threads[j] = thread(check_divisibility, start, end, i);
            }
            for (int j = 0; j < thread_count; ++j) {
                threads[j].join();
            }
            if (dividers == 1) {
                cout << i << " "<<endl;
            }
        }
    }
}

int main() {
    cout << "Zadanie 1:" << endl;
    zad1();
    cout << "Zadanie 2:" << endl;
    zad2();
    cout << "Zadanie 3:" << endl;
    zad3();
    cout << "Zadanie 4:" << endl;
    zad4();
    cout << "Zadanie 5:" << endl;
    zad5();
    cout << "Zadanie dodatkowe:" << endl;
    zad_dodatkowe(1000);

    return 0;
}